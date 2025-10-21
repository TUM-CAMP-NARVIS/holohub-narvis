#
# Place the license header here
#
import os
import logging
from argparse import ArgumentParser
from typing import Any, Dict, List, Optional, Callable, Tuple, Union
import queue
import threading

import holoscan as hs

from holoscan.gxf import Entity
from holoscan.logger import LogLevel, set_log_level

from holoscan.resources import CudaStreamPool, UnboundedAllocator, BlockMemoryPool

from holoscan.conditions import AsynchronousCondition, AsynchronousEventState, BooleanCondition, CountCondition, PeriodicCondition, MessageAvailableCondition, DownstreamMessageAffordableCondition
from holoscan.core import Application, ConditionType, IOSpec, Operator, OperatorSpec, Tracker
# from holoscan.schedulers import EventBasedScheduler, GreedyScheduler, MultiThreadScheduler

from holoscan.operators import HolovizOp

from holohub.nv_video_decoder import NvVideoDecoderOp


import zenoh

from tcnart.network.discovery import find_camera_sensors, build_channel_configs
from tcnart.network.receiver import resolve_stream_descriptors, start_all_receivers
from tcnart.serialization.cdr_serialization import decode_raw_message, encode_raw_message
from tcnart.serialization.error import MessageError
from tcnart.core.semantic_type import SemanticType
from tcnart.schema.messages.rpc import NullRequest
from tcnart.schema.messages.service_controller import DeviceContextReply
from tcnart.schema.messages.stream import StreamDescriptorMessage
from tcnart.schema.messages.video import VideoStreamMessage
from tcnart.schema.messages.common import InvalidMessage
from tcnart.schema.types.primitives import CameraModel
from tcnart.schema.types.transform import RigidTransform
from tcnart.core.dataflow import StreamConfig
from tcnart.core.frames import FrameAnnotation

log = logging.getLogger(__name__)


class StatsOp(Operator):
    """Print common streaming statistics"""

    def __init__(self, app, *args, **kwargs):
        self.encode_latency = []
        self.decode_latency = []
        self.jitter_time = []
        self.fps = []
        self.first_frame_ignored = False
        self._logger = logging.getLogger(__name__)
        super().__init__(app, *args, **kwargs)

    def setup(self, spec):
        spec.input("input")

    def compute(self, op_input, op_output, context):
        _ = op_input.receive("input")
        if not self.first_frame_ignored:
            self.first_frame_ignored = True
            return

        # Check if metadata exists before accessing it
        if hasattr(self, "metadata"):
            self.encode_latency.append(self.metadata.get("video_encoder_encode_latency_ms", 0))
            self.decode_latency.append(self.metadata.get("video_decoder_decode_latency_ms", 0))
            self.jitter_time.append(self.metadata.get("jitter_time", 0))
            self.fps.append(self.metadata.get("fps", 0))

    def stop(self):
        self._logger.info(
            f"Encode Latency (ms) (min, max, avg): {min(self.encode_latency):.3f}, {max(self.encode_latency):.3f}, {sum(self.encode_latency) / len(self.encode_latency):.3f}"
        )
        self._logger.info(
            f"Decode Latency (ms) (min, max, avg): {min(self.decode_latency):.3f}, {max(self.decode_latency):.3f}, {sum(self.decode_latency) / len(self.decode_latency):.3f}"
        )
        self._logger.info(
            f"Jitter Time (ms) (min, max, avg): {min(self.jitter_time):.3f}, {max(self.jitter_time):.3f}, {sum(self.jitter_time) / len(self.jitter_time):.3f}"
        )
        self._logger.info(
            f"FPS (min, max, avg): {min(self.fps):.3f}, {max(self.fps):.3f}, {sum(self.fps) / len(self.fps):.3f}"
        )


class CdrDecoderOp(Operator):
    """Decode CDR Payload Baseclass.

    This operator has 1 input and 1 output port:
        input:  "in"
        output: "out"

    The data from each input is multiplied by a user-defined value.

    """

    def __init__(self,
                 fragment: Any,
                 type_class: Any,
                 result_factory: Callable,
                 source: str,
                 stream_index: int,
                 semantic_type: SemanticType,
                 annotations: Dict[str, FrameAnnotation],
                 *args,
                 **kwargs):

        self.type_class = type_class
        self.source = source
        self.stream_index = stream_index
        self.semantic_type = semantic_type
        self.annotations = annotations
        self.result_factory = result_factory if result_factory else lambda m, tn, d: d

        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("input")
        spec.output("output")

    def compute(self, op_input, op_output, context):
        value = op_input.receive("in")

        type_name = self.metadata.get("cdr_type_name", None)

        if type_name is None:
            return

        self.metadata.set("source", self.source)
        self.metadata.set("stream_index", self.stream_index)
        self.metadata.set("semantic_type", self.semantic_type)

        for k,v in self.annotations.items():
            self.metadata.set(k, v)

        # decode using the provided type_class
        try:
            # allocates buffers in library ...
            msg = self.type_class.deserialize(value)
        except MessageError as e:
            log.exception(e)
            msg = InvalidMessage()

        result = self.result_factory(self.metadata, type_name, msg)
        op_output.emit(result, "out")




class ZenohSubscriberOp(Operator):
    """Simple zenoh subscriber.

    On each tick, it transmits a received message to the "out" port.

    **==Named Outputs==**

        out : bytes
        the received payload.
    """

    def __init__(
        self,
        fragment: Any,
        session: Any,
        topic: str,
        *args,
        **kwargs,
    ):
        self.session = session
        self.topic = topic

        self.subscriber = None
        self.pool = None

        self.async_cond_ = AsynchronousCondition(fragment, name="async_cond")
        self.buffer = queue.Queue()

        # Need to call the base class constructor last
        super().__init__(fragment, self.async_cond_, *args, **kwargs)

    def on_receive(self, sample: Any):
        """Function to be supplied as callback

        When the condition's event_state is EVENT_WAITING, set to EVENT_DONE. This function will
        only exit once the condition is set to EVENT_NEVER.
        """
        if self.async_cond_.event_state == AsynchronousEventState.EVENT_NEVER:
            return

        if sample is None or sample.payload is None or sample.attachment is None:
            log.warning("received incomplete sample.")
            return

        try:
            payload = sample.payload.to_bytes()
            type_name = sample.attachment.to_string()
        except Exception as e:
            log.exception(e)
            type_name = None
            payload = None

        if payload is not None and type_name is not None:
            # how does ts relate to fragment.scheduler().clock.timestamp()?
            self.buffer.put((type_name, payload))

            if self.async_cond_.event_state == AsynchronousEventState.EVENT_WAITING:
                self.async_cond_.event_state = AsynchronousEventState.EVENT_DONE

    def setup(self, spec: OperatorSpec):
        spec.output("output")

    def start(self):
        self.subscriber = self.session.declare_subscriber(self.topic, self.on_receive)

    def compute(self, op_input, op_output, context):
        scheduler = self.fragment.scheduler()
        clock = scheduler.clock
        ts = clock.timestamp()

        type_name, block = self.buffer.get()

        self.async_cond_.event_state = AsynchronousEventState.EVENT_WAITING

        self.metadata.set("cdr_type_name", type_name)

        op_output.emit(block, "out", acq_timestamp=ts)

    def stop(self):
        self.async_cond_.event_state = AsynchronousEventState.EVENT_NEVER
        if self.subscriber is not None:
            self.subscriber.undeclare()
            self.subscriber = None



class PingRxOp(Operator):
    """Simple receiver operator.

    This is an example of a native operator with one input port.
    On each tick, it receives an integer from the "in" port.

    **==Named Inputs==**

        in : any
            A received value.
    """

    def __init__(self, fragment, *args, **kwargs):
        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("input")

    def compute(self, op_input, op_output, context):
        value = op_input.receive("in")
        print(f"Received bytes: {len(value)}", self.metadata.keys())


class App(hs.core.Application):
    def compose(self):
        # Add your operators here
        print("Starting TCN Test Receiver")

        zenoh_config = self.kwargs("zenoh")

        topic_prefix = zenoh_config.get("topic_prefix")
        capture_node = zenoh_config.get("capture_node")
        zenoh_config_file = zenoh_config.get("zenoh_config_file")

        zenoh.init_log_from_env_or("warn")
        self.session = zenoh.open(zenoh.Config.from_json5(open(zenoh_config_file).read()))

        find_cameras_topic = f"{topic_prefix}/{capture_node}/rpc/sensor/*/describe"
        print(f"Find cameras: {find_cameras_topic}")

        cameras = find_camera_sensors(self.session, find_cameras_topic)
        channels_config, channel_calibration, channel_poses = build_channel_configs(cameras)
        stream_config = resolve_stream_descriptors(
            topic_prefix, None, channel_calibration, channel_poses, channels_config, self.session
        )
        stream_keys = list(sorted(stream_config.keys()))


        def cb_decoder(meta, type_name, msg):
            # is a video message, so return the raw image-bytes (typically bit/bytestream)
            return msg.image

        for stream_index, stream_name in enumerate(stream_keys):
            config = stream_config[stream_name]
            topic = config.descriptor.stream_topic
            subscriber = ZenohSubscriberOp(self, self.session, topic)
            deserializer = CdrDecoderOp(self, VideoStreamMessage, cb_decoder, stream_name, stream_index,
                                   SemanticType.from_identifier(config.descriptor.buffer_info.semantic_type),
                                   config.annotations)
            printer = PingRxOp(self)

            decoder = NvVideoDecoderOp(
                self,
                name="nv_decoder",
                allocator=UnboundedAllocator(self, name="video_decoder_pool"),
                **self.kwargs("decoder"),
            )

            stats = StatsOp(self, name="stats")

            self.add_flow(subscriber, deserializer, {('output', 'input')})
            self.add_flow(deserializer, printer, {('output', 'input')})
            self.add_flow(deserializer, decoder, {('output', 'input')})
            self.add_flow(decoder, stats, {("output", "input")})



def main(config_file=None):
    # make configurable or use holoscan debug level here too
    logging.basicConfig(level=logging.DEBUG)

    app = App()
    app.config(config_file)
    app.run()


if __name__ == "__main__":


    parser = ArgumentParser(description="ARTEKMED Holoscan Client.")

    parser.add_argument(
        "-c",
        "--config",
        default="none",
        help=("Set config path to override the default config file location"),
    )

    args = parser.parse_args()

    if args.config == "none":
        config_file = config_file = os.path.join(os.path.dirname(__file__), "tcn_receiver_test.yaml")
    else:
        config_file = args.config

    main(config_file=config_file)

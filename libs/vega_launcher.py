import vega
from vega import vegas_pb2 as vegas
import logging
import vega


def is_vega_launched()->bool:
    cmd = vega.Command("vegapy", "read_capability")
    vega.send_cmd(cmd)
    return cmd.err.code != 2


def load_vega():
    if not is_vega_launched():
        logging.info(f'vega py {vega.__version__}  on {vega.__vega_lib_base_infer__}/{vega.__vega_lib_build_type__}')
        lc = vega.Launcher()
        lc.setLogLevel(2, vlog=None)  # ignore all non-error logs from vega
        lc.setDevice(0)  # select device, change to your device id if required
        lc.launch()
        # read vega capability, the content will be proto stream
        cmd = vega.Command("vegapy", "read_capability")
        vega.send_cmd(cmd)
        assert cmd.err.ok()
        # because it contains invalid utf-8 char, so get binary of proto
        s = cmd.get_bin("result")
        cap = vegas.VegaCapability()
        cap.ParseFromString(s)
        logging.info(f'vega version: {cap.version}, device: {cap.deviceId}, platform: {cap.platform}, deviceModel: {cap.deviceModel}')
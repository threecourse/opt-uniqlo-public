import logging
import datetime

class Logger:

    def __init__(self):
        self.g_logger = logging.getLogger("general")
        self.r_logger = logging.getLogger("result")
        stream_handler = logging.StreamHandler()
        file_general_handler = logging.FileHandler("../model/general.log")
        file_result_handler = logging.FileHandler("../model/result.log")
        # formatter = logging.Formatter('my-format')
        if len(self.g_logger.handlers) == 0:
            self.g_logger.addHandler(stream_handler)
            self.g_logger.addHandler(file_general_handler)
            self.g_logger.setLevel(logging.INFO)
            self.r_logger.addHandler(stream_handler)
            self.r_logger.addHandler(file_result_handler)
            self.r_logger.setLevel(logging.INFO)

    def now_ymdHMS(self):
        return str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def to_ltsv(cls, ordered_dict):
        return "\t".join(["{}:{}".format(key, value) for key, value in ordered_dict.items()])

    def result(self, message):
        self.r_logger.info(message)

    def result_ltsv(self, ordered_dict):
        self.result(self.to_ltsv(ordered_dict))

    def result_ltsv_time(self, ordered_dict):
        self.result("time:{}\t".format(self.now_ymdHMS()) + self.to_ltsv(ordered_dict))

    def info(self, message):
        self.g_logger.info("[{}] - {}".format(self.now_ymdHMS(), message))


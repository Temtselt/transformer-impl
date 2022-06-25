import datetime
from rich import print


class Logger:
    @staticmethod
    def log(tag: str, message: str, level: str = "LOG"):
        print(f"{datetime.datetime.now()} | {level:7s} | {tag}: {message}")

    @staticmethod
    def logv(tag: str, message: str):
        Logger.log(tag, message, "VERBOSE")

    @staticmethod
    def logd(tag: str, message: str):
        Logger.log(tag, message, "DEBUG")

    @staticmethod
    def logi(tag: str, message: str):
        Logger.log(tag, message, "INFO")

    @staticmethod
    def logw(tag: str, message: str):
        Logger.log(tag, message, "WARN")

    @staticmethod
    def loge(tag: str, message: str):
        Logger.log(tag, message, "ERROR")


if __name__ == "__main__":
    for l in (
        Logger.log,
        Logger.logv,
        Logger.logd,
        Logger.logi,
        Logger.logw,
        Logger.loge,
    ):
        l("foo", "bar")

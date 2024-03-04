import inspect
from loguru import logger

class LoggingManager():
    def __prepare_callstack(self):
        current_callstack_list = []

        for frame_info in reversed(inspect.stack()):
            if("karaburma" in frame_info.filename and "<module>" not in frame_info.function and "logging_manager" not in frame_info.filename):
                current_callstack_list.append("method '{}' was called from '{}'".format(frame_info.function, frame_info.filename.split("\\")[-1]))

        return current_callstack_list

    @staticmethod
    def log_number_arguments_error(excpected_value, actual_value):
        current_callstack_list = LoggingManager().__prepare_callstack()
        callstack_message = "Callstack: " + "\n".join(current_callstack_list)
        error_message = f"{current_callstack_list[-1]} expects {excpected_value} arguments but was {actual_value}"
        logger.error(error_message + "\n" + callstack_message)

        return error_message

    @staticmethod
    def log_error(orig_error_message):
        current_callstack_list = LoggingManager().__prepare_callstack()
        callstack_message = "Callstack: " + "\n".join(current_callstack_list)
        logger.error(orig_error_message  + "\n" + callstack_message)

    @staticmethod
    def log_information(message):
        logger.info(message)
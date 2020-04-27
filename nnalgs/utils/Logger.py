import logging
import logging.config


def get_logger(name, msg):
    """
    :param name: string
    :param msg: DEBUG, INFO, WARNING, ERROR
    :return: Logger() instance
    """
    level = {"DEBUG": logging.DEBUG,
             "INFO": logging.INFO,
             "WARNING": logging.WARNING,
             "ERROR": logging.ERROR}
    logging.basicConfig(level=level[msg], format='== %(name)s == %(created)f %(levelname)s:\t%(message)s')
    logger = logging.getLogger(name)

    return logger

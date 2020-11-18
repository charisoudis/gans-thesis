from src.utils.command_line_logger import CommandLineLogger


def main():
    logger = CommandLineLogger(log_level='debug')
    # logger.log_format = "> %(log_color)s%(message)s%(reset)s"
    logger.info('execution started')


if __name__ == '__main__':
    main()

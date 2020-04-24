# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author  :  Yan Errol 
@Describe:  主函数入口
@Evn     :
@Date    :  2019-07-08  19:09
'''

import argparse
import logging


def file_exists(path):
    '''
    判断文件是否存在
    :param path: 文件路径
    :return: path
    '''
    if not os.path.isfile(path):
        raise argparse.ArgumentTypeError("文件 %s 不存在!" % path)
    return path


def path_exists(path):
    '''
    判断文件夹是否存在
    :param path: 文件夹路径
    :return: path
    '''
    if not os.path.isdir(path):
        raise argparse.ArgumentTypeError("目录 %s 不存在!" % path)
    return path

def parser_args():
    '''
    解析命令行参数
    :return:
    '''
    parser = argparse.ArgumentParser(description="收入预测模型训练程序")
    parser.add_argument("-v", "--version", action="version", version="2.0 Author:Yan Errol ", help="打印版本信息")

    subparsers = parser.add_subparsers(title="操作命令", dest="command")
    train_parser = subparsers.add_parser("train", help="训练模型")
    train_parser.add_argument("-s", "--steps", help="训练轮数", type=int, required=True)

    export_parser = subparsers.add_parser("predict", help="模型预测")
    return parser.parse_args()

def main():
    '''
    主函数调用
    :return:
    '''

    logger = logging.getLogger('income_predict')
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(filename="./income_predict.log")
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    args = parser_args()
    if args.command is None:
        print("usage: python3 main.py [-h] [-v] {train,predict} ...")

    else:
        try:
            if args.command == "train":
                logger.info("------Beginning Train------")
                logger.debug('This is a customer debug message')
                logger.warning('This is a customer warning message')
                logger.error('This is an customer error message')
                logger.critical('This is a customer critical message')
                print("Training...")
                logger.info("------Training Over------")
            elif args.command == "server":
                logger.info("------Begining Predict------")
                print("Predict ...")
                logger.info("------Predict Over------")
        except Exception as e:
            print("error", e)
            traceback.print_exc()
            logging.exception(e)
            logger.critical(e)
            logger.critical("error Exit")
        else:
            print(r"All_is_Over")
        logger.info("--------All_is_Over--------")


if __name__ == "__main__":
    main()
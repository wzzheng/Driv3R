from driv3r.train_driv3r import get_args_parser, train

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    train(args)
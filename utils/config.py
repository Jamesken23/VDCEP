import argparse


def create_parser():
    parser = argparse.ArgumentParser(description='-----A vulnerability detection framework by focusing on critical execution paths (VDCEP) --PyTorch ')
    
    # Architecture
    parser.add_argument('--embedding_dim', default=400, type=int)
    parser.add_argument('--vocab_size', default=1000, type=int)
    
    # loader
    parser.add_argument('--is_balanced', default=True)


    # Data
    parser.add_argument('--SC_Type', default="RE", choices=["RE", "TP", "OF", "DE"])
    parser.add_argument('--num_classes', type=int, default="2", help='number of class')
    parser.add_argument('--max_setence_length', type=int, default="2000", help='Max length of a setence')
    
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')

    
    # Optimization
    parser.add_argument('--dropout', default=0.5, type=float, help='ratio of dropout (default: 0)')
    parser.add_argument('--epochs', type=int, default="100", help='number of total training epochs')
    parser.add_argument('--optim', default="sgd", type=str, metavar='TYPE', choices=['sgd', 'adam'])

    # Log and save
    parser.add_argument('--print-freq', default=20, type=int, metavar='N', help='display frequence (default: 20)')
    parser.add_argument('--save-freq', default=0, type=int, metavar='EPOCHS', help='checkpoint frequency(default: 0)')
    parser.add_argument('--log_dir', default='./logs', type=str, metavar='DIR')
    parser.add_argument('--save-dir', default='./checkpoints', type=str, metavar='DIR')


    return parser.parse_args()


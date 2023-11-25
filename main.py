from datetime import datetime
import argparse
import sys

sys.path.append('C:\\repos\\vct-ml-soccer')

from src.dataset import VideoAction
from src.multimodal import Transcript
from src.classifier import CelebraPredict, CelebraTrainer, CelebraTuner
from configs import log_conf

LOGGER = log_conf.getLogger(__name__)


def main(args):
    if args.option == 'dataset':
        VideoAction(output_path=args.output_path,
                    orig_annots=args.annots,
                    orig_task=args.task,
                    padding=args.padding,
                    action=args.action,
                    num_videos=args.num_videos,
                    level=args.log).pipeline()

    if args.option == 'transcript':
        Transcript(video_fn=args.video_fn,
                   output_path=args.output_path).run()

    if args.option == 'trainer':
        CelebraTrainer(num_epochs=args.epochs,
                       batch_size=args.b_size,
                       learning_rate=args.learning_rate,
                       checkpoint_freq=args.checkpoint_frequency,
                       level=args.log).run_for_epochs()

    if args.option == 'predict':
        CelebraPredict(level=args.log).predict(x=args.input, plot=args.plot)

    if args.option == 'tuner':
        CelebraTuner(num_trials=args.trials,
                     max_epochs_per_trial=args.epochs,
                     level=args.log).run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(dest='option')

    # dataset option
    dataset = subparser.add_parser('dataset', help='From the SoccetNet datatet("https://www.soccer-net.org/") '
                                                   'downloaded and extracted in the local folder, creates short '
                                                   'videos around a given action from the annotations')
    dataset.add_argument('-op', '--output_path', default=f'videos_{datetime.now().strftime("%y%m%d%H%M%S")}',
                         help='destination folder of the dataset')
    dataset.add_argument('-ap', '--annots', default='france_ligue-1', type=str,
                         help='annotations used to generate the output videos')
    dataset.add_argument('-ot', '--task', default='actionspotting', type=str, help='dataset task')
    dataset.add_argument('-p', '--padding', default=20, type=int, help='padding')
    dataset.add_argument('-a', '--action', default='Goal', type=str, help='action')
    dataset.add_argument('-n', '--num_videos', default=20, type=int)
    dataset.add_argument('-l', '--log', type=str, choices=['NOTSET', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                         default='INFO')

    # transcript option
    transcript = subparser.add_parser('transcript', help='Takes an mp4 file and gets the transcripted text of the file')
    transcript.add_argument('-v', '--video_fn', type=str, help='absolute video path from where get the transcript')
    transcript.add_argument('-o', '--output_path', type=str, help='relative output path for the transcript')
    transcript.add_argument('-l', '--log', type=str,
                            choices=['NOTSET', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                            default='INFO')

    # classifier - train
    celebtrainer = subparser.add_parser('trainer', help='Trainer for frames of futbol match. Classifies in between'
                                                        ' next classes: ["banquillo", "celebration", "general",'
                                                        ' "medium", "short", "trainer", "tribuna"]')
    celebtrainer.add_argument('-e', '--epochs', type=int, default=5, help='mum epochs')
    celebtrainer.add_argument('-b', '--b_size', type=int, default=4, help='batch_size')
    celebtrainer.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='learning_rate')
    celebtrainer.add_argument('-f', '--checkpoint_frequency', type=int, default=1, help='checkpoint_frequency')
    celebtrainer.add_argument('-l', '--log', type=str,
                              choices=['NOTSET', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                              default='DEBUG')

    # classifier - predict
    celebpredict = subparser.add_parser('predict', help='Celebra classifier predictor')
    celebpredict.add_argument('-i', '--input', type=str, help='input image')
    celebpredict.add_argument('-p', '--plot', action='store_true', help='plot')
    celebpredict.add_argument('-l', '--log', type=str,
                              choices=['NOTSET', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                              default='DEBUG')

    # classifier - tuner
    celebtuner = subparser.add_parser('tuner', help='Hyperparameters tunning')
    celebtuner.add_argument('-t', '--trials', type=int, default=10,
                            help='Maximum n umber of trials taking into account the chosen config')
    celebtuner.add_argument('-e', '--epochs', type=int, default=30,
                            help='Maximum number of epochs per trial as stop criteria')
    celebtuner.add_argument('-l', '--log', type=str,
                            choices=['NOTSET', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                            default='DEBUG')

    args = parser.parse_args()
    main(args)

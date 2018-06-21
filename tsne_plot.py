from sklearn.manifold import TSNE
import os
import numpy as np
from src.parameters import config_select
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1,
                rc={"lines.linewidth": 2.5})

phonemes = ['S', 'EnEn', 'O', 'nvc', 'N', 'j', 'in', 'y', '@n', 'i', 'MM', 'oU^', 'SN', 'aI^', 'an', 'AU^', 'rr',
            'ANAN', '@', 'a', 'vc', 'iNiN', 'eI^', 'UN', 'u', 'E', 'ONE']  # perplexity 30


def plot_tsne_profess_all(embeddings, labels, dense=False):
    tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=5000)

    for p in phonemes:
        index_teacher = np.where(labels == p+"_teacher")[0]
        index_student = np.where(labels == p+"_student")[0]
        index_test = np.where(labels == p+"_extra_test")[0]
        # plot t_sne for teacher and student
        try:
            tsne_results = tsne.fit_transform(np.vstack((embeddings[index_teacher, :],
                                                         embeddings[index_student, :],
                                                         embeddings[index_test, :])))
            plt.figure()

            plt.scatter(tsne_results[:len(index_teacher), 0],
                        tsne_results[:len(index_teacher), 1],
                        label=p+" Professional")

            plt.scatter(tsne_results[len(index_teacher):len(index_teacher)+len(index_student), 0],
                        tsne_results[len(index_teacher):len(index_teacher)+len(index_student), 1],
                        label=p+" Amateur\ntrain val",
                        marker='v')

            plt.scatter(tsne_results[len(index_teacher) + len(index_student):, 0],
                        tsne_results[len(index_teacher) + len(index_student):, 1],
                        label=p+" Amateur\ntest",
                        marker='+')

            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                       ncol=3, mode='expand', borderaxespad=0.)

            dense_str = "dense_all" if dense else "all"
            plt.savefig(os.path.join('./figs/overall_quality/'+dense_str+'/', p+'.png'),
                        bbox_inches='tight')
        except:
            pass


def run_tsne_plot(embedding_path, dense):
    config = [1, 0]
    prefix = '_2_class_teacher_student'
    model_name = config_select(config) + prefix
    dense_str = "dense_32__dense" if dense else ""
    embedding_profess = np.load(
        os.path.join(embedding_path, model_name + '_embedding_overall_quality' + dense_str + '_all0.npy'))
    labels = np.load(os.path.join(embedding_path, model_name + '_embeddings_labels' + dense_str + '_all.npy'))
    plot_tsne_profess_all(embedding_profess, labels, dense=dense)


if __name__ == '__main__':

    import argparse

    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser(description="Plot the t-SNE visualization.")

    parser.add_argument("-e",
                        "--embedding_path",
                        type=str,
                        help="Type the embedding path")

    parser.add_argument("--dense",
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default="False",
                        help="choose the dense.")

    args = parser.parse_args()

    run_tsne_plot(embedding_path=args.embedding_path,
                  dense=args.dense)
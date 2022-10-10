from io import StringIO

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns


def main():
    plot_str = '''pos,value
    NOUN,0.2879
    DET,0.1942
    PUNCT,0.4465
    ADP,0.3188
    VERB,0.2969
    ADJ,0.2886
    ADV,0.2925
    NUM,0.1585
    CCONJ,0.3687
    PRON,0.2788
    '''

    sns.set_style('whitegrid')
    plt.rcParams.update({'font.size': 12})
    plt.rcParams['figure.figsize'] = [9, 0.5]
    df1 = pd.read_csv(StringIO(plot_str))
    ax = sns.boxplot(x=df1.value, color='lightgrey')
    for line in ax.lines:
        print(line.get_ydata())

    plt.annotate('DET', (0.1946, -0.05))
    plt.annotate('PUNCT', (0.4466 - 0.027, -0.05))
    plt.annotate('CCONJ', (0.3688 - 0.027, -0.05))
    plt.annotate('NUM', (0.1592, -0.05))
    plt.title('Attribution Intensity by Part-of-Speech')
    plt.savefig('pos.pdf', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()

from dataclasses import dataclass
import nltk

from admin.nlp.samsung_report.services import Service


@dataclass
class Entity:
    context: str
    fname: str
    target: str

    @property
    def context(self) -> str: return self._context

    @context.setter
    def context(self, context): self._context = context

    @property
    def fname(self) -> str: return self._fname

    @fname.setter
    def fname(self, fname): self._fname = fname

    @property
    def target(self) -> str: return self._target

    @target.setter
    def target(self, target): self._target = target





class Controller:
    def __init__(self):
        self.entity = Entity()

    def download_dictionary(self):
        nltk.download('all')

    def data_analysis(self):
        self.entity.fname = 'kr-Report_2018.txt'
        self.entity.context = r'C:\Users\AIA\PycharmProjects\djangoProject\admin\nlp\samsung_report'
        Service.extract_tokens(self.entity)
        Service.extract_hangeul()
        Service.conversion_token()
        Service.compound_noun()
        self.entity.fname = 'stopwords.txt'
        Service.extract_stopword(self.entity)
        Service.filtering_text_with_stopword()
        Service.frequent_text()
        self.entity.fname = 'D2Coding.ttf'
        Service.draw_wordcloud(self.entity)


if __name__ == '__main__':
    app = Controller()
    # app.download_dictionary()
    app.data_analysis()
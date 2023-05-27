import csv
from translate import Translator
from googletrans import Translator as Gtranslator


class CSVTranslator:
    """ A class for translating CSV files using translate library.

    Args:
        input_file (str): The path to the input CSV file.
        output_file (str): The path to the output CSV file.

    Attributes:
        input_file (str): The path to the input CSV file.
        output_file (str): The path to the output CSV file.
    """

    def __init__(self, input_file: str, output_file: str):
        self.input_file = input_file
        self.output_file = output_file

    def translate_csv(self, src_lang: str, dest_lang: str):
        """ Translates the content of the input CSV file using translate library and saves the translated data to the output CSV file.

        Args:
            src_lang (str): The source language code.
            dest_lang (str): The target language code.
        """
        translator = Translator(from_lang=src_lang, to_lang=dest_lang)
        with open(self.input_file, 'r') as input_file, open(self.output_file, 'w', encoding='utf-8-sig') as output_file:
            reader = csv.reader(input_file)
            writer = csv.writer(output_file)
            for row in reader:
                translated_row = []
                for cell in row:
                    try:
                        if cell and not cell.isnumeric():
                            translation = translator.translate(cell)
                            if translation.lower() == cell:
                                translation = Gtranslator().translate(
                                    cell, src=src_lang, dest=dest_lang).text.lower()
                        else:
                            translation = cell
                        translated_row.append(translation)
                    except Exception as e:
                        print(
                            f"Cell {cell} can't translated because error: {e}")
                        translated_row.append('')
                writer.writerow(translated_row)


if __name__ == "__main__":
    translated = CSVTranslator(
        "datasets/sentiment_words.csv", "datasets/translated_sentiment_words.csv")
    translated.translate_csv('en', 'pt')

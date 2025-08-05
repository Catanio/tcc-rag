from translator import CorpusTranslator

def translate_example():
    model_name = "facebook/nllb-200-distilled-600M"
    translator = CorpusTranslator(model_name, batch_size=4)
    
    test_phrases = [
        "Olá, como você está?",
        "O gato está em cima do tapete.",
        "O rato roeu a roupa do rei de Roma.",
        "Este é um teste de tradução automática."
    ]
    translated = translator.translate_batch(test_phrases)
    print("\nResultados da tradução:")
    for original, translation in zip(test_phrases, translated):
        print(f"Original (PT): {original}")
        print(f"Tradução (EN): {translation}")
        print("-" * 50)

if __name__ == '__main__':
    translate_example()
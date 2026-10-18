import tiktoken
from Task4 import train_bpe

if __name__ == '__main__':
    # 1. Wikipedia "Aşı (botanik)" Sayfasından İlk Paragraflar ve İngilizce Çevirisi
    tr_text = (
        "Aşılama, bitkilerde uygulanan eşeysiz bir üreme biçimidir. Çoğaltılması istenilen çeşitten, "
        "bir gözün ya da aşı kalemi adı verilen bir dal parçasının anaç adı verilen diğer bir bitki "
        "üzerine yerleştirilip tutturulmasıdır.\n\n"
        "Meyve çeşitleri, genel olarak, tohumla üretildiklerinde çeşit niteliklerini kaybederek, "
        "yabanileşmeye doğru yönelmektedirler. Üreticiler, kalıtsal özelliklerin korunumu için, "
        "aşı yöntemi ile üretmek zorunda kalırlar. Üretilmesi istenilen, nitelikli, bol verimli "
        "ve hastalıklara dayanıklı meyve çeşitlerini, aşılamak yoluyla çoğaltmak olanağı sağlanmaktadır.\n\n"
        "Bağ-bahçe tarımında kullanılan birçok aşılama biçimi vardır. Bunlardan en çok kullanılanları, "
        "göz ve kalem aşılarıdır."
    )

    en_text = (
        "Grafting is a form of asexual reproduction applied in plants. It involves placing and "
        "attaching a bud or a branch piece called a scion from the desired variety onto another "
        "plant called a rootstock.\n\n"
        "Fruit varieties, in general, tend to revert to their wild forms and lose their varietal "
        "characteristics when propagated by seed. Growers are forced to propagate through grafting "
        "to preserve genetic traits. Grafting makes it possible to multiply desirable, high-quality, "
        "high-yielding, and disease-resistant fruit varieties.\n\n"
        "There are many forms of grafting used in orchard and vineyard agriculture. The most "
        "commonly used among them are bud and scion grafting."
    )

    # 2. GPT-4 Tokenizer'ı (tiktoken, "cl100k_base")
    enc_gpt4 = tiktoken.get_encoding("cl100k_base")

    tr_gpt4_tokens = enc_gpt4.encode(tr_text)
    en_gpt4_tokens = enc_gpt4.encode(en_text)

    tr_word_count = len(tr_text.split())
    en_word_count = len(en_text.split())

    custom_encode, _, _, _ = train_bpe(verbose=False)
    tr_custom_tokens = custom_encode(tr_text)

    print(f"GPT-4 Tokenizer on English: (token per word): {len(en_gpt4_tokens) / en_word_count:.2f}")
    print(f"GPT-4 Tokenizer on Turkish: (token per word): {len(tr_gpt4_tokens) / tr_word_count:.2f}")
    print(f"Our Turkish-trained Tokenizer: {len(tr_custom_tokens) / tr_word_count:.2f}")

"""
GPT-4 Tokenizer on English: (token per word): 1.33
GPT-4 Tokenizer on Turkish: (token per word): 3.34
Our Turkish-trained Tokenizer: 7.31

tutturulmasıdır (7 token): ['t', 'utt', 'ur', 'ul', 'ması', 'd', 'ır']
yönelmektedirler (7 token): ['y', 'ö', 'nel', 'mek', 'ted', 'ir', 'ler']
üretildiklerinde (6 token): ['ü', 'ret', 'ild', 'ik', 'ler', 'inde']
yabanileşmeye (6 token): ['y', 'aban', 'ile', 'ş', 'm', 'eye']
"""
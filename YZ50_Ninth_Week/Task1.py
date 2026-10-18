
if __name__ == '__main__':
    turkish_pangram = "Pijamalı hasta, yağız şoföre çabucak güvendi."
    print("Unicode kod noktası:")
    print({l: ord(l) for l in turkish_pangram})

    print("UTF-8 byte'ları")
    print({l: l.encode("utf-8") for l in turkish_pangram})

"""
2-byte UTF-8 representation olan harfler
 'ı': b'\xc4\xb1',
 'ğ': b'\xc4\x9f',
 'ş': b'\xc5\x9f',
 'ö': b'\xc3\xb6',
 'ç': b'\xc3\xa7',
 'ü': b'\xc3\xbc',
"""

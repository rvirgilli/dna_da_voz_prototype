from voice_verificator import VoiceVerificator

if __name__ == '__main__':
    vv = VoiceVerificator()

    #set reference file
    vv.references('./files/examples/lucas_ref.wav')

    #check against multiple verification files (True -> same person / False -> different persons)
    print(vv.verification('./files/examples/lucas_ver1.wav'))
    print(vv.verification('./files/examples/lucas_ver2.wav'))
    print(vv.verification('./files/examples/rafaello_ver1.wav'))
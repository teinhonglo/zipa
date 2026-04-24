import local.phonecode_tables as phonecode_tables
import argparse

def translate_string(s, d):
    '''(tl,ttf)=translate_string(s,d):
    Translate the string, s, using symbols from dict, d, as:
    1. Min # untranslatable symbols, then 2. Min # symbols.
    tl = list of translated or untranslated symbols.
    ttf[n] = True if tl[n] was translated, else ttf[n]=False.
'''
    N = len(s)
    symcost = 1    # path cost per translated symbol
    oovcost = 10   # path cost per untranslatable symbol
    maxsym = max(len(k) for k in d.keys())  # max input symbol length
    # (pathcost to s[(n-m):n], n-m, translation[s[(n-m):m]], True/False)
    lattice = [ (0,0,'',True) ]
    for n in range(1,N+1):
        # Initialize on the assumption that s[n-1] is untranslatable
        lattice.append((oovcost+lattice[n-1][0],n-1,s[(n-1):n],False))
        # Search for translatable sequences s[(n-m):n], and keep the best
        for m in range(1,min(n+1,maxsym+1)):
            if s[(n-m):n] in d and symcost+lattice[n-m][0] < lattice[n][0]:
                lattice[n] = (symcost+lattice[n-m][0],n-m,d[s[(n-m):n]],True)
    # Back-trace
    tl = []
    translated = []
    n = N
    while n > 0:
        tl.append(lattice[n][2])
        translated.append(lattice[n][3])
        n = lattice[n][1]
    return((tl[::-1], translated[::-1]))

def attach_tones_to_vowels(il, tones, vowels, searchstep, catdir):
    '''Return a copy of il, with each tone attached to nearest vowel if any.
    searchstep=1 means search for next vowel, searchstep=-1 means prev vowel.
    catdir>=0 means concatenate after vowel, catdir<0 means cat before vowel.
    Tones are not combined, except those also included in the vowels set.
    '''
    ol = il.copy()
    v = 0 if searchstep>0 else len(ol)-1
    t = -1
    while 0<=v and v<len(ol):
        if (ol[v] in vowels or (len(ol[v])>1 and ol[v][0] in vowels)) and t>=0:
            ol[v]= ol[v]+ol[t] if catdir>=0 else ol[t]+ol[v]
            ol = ol[0:t] + ol[(t+1):]  # Remove the tone
            t = -1 # Done with that tone
        if v<len(ol) and ol[v] in tones:
            t = v
        v += searchstep
    return ol

# 定義轉換函數，將每個 word 的音標黏合
def arpabet_to_ipa_conversion(x):
    '''Convert ARPABET symbol X to IPA'''
    print(x)
    x = " ".join(x)
    (il,ttf)=translate_string(x, phonecode_tables._arpabet2ipa)
    ol = attach_tones_to_vowels(il, phonecode_tables._ipa_stressmarkers,
                              phonecode_tables._ipa_vowels,-1,-1)
    
    result = "".join("".join(ol).split())
    return result

# 讀取並轉換檔案內容
def convert_file(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    output_lines = []
    for line in lines:
        # 將行轉換為所需格式，取得 uttid 並轉換音標
        uttid = line.split()[0]
        phone_list = eval(" ".join(line.split()[1:]))
        ipa_converted = " ".join([arpabet_to_ipa_conversion(w) for w in phone_list])
        output_lines.append(f"{uttid} {ipa_converted}")

    with open(output_file, 'w') as f:
        for line in output_lines:
            f.write(line + '\n')

# 主程式
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert ARPAbet to IPA.")
    parser.add_argument('--input_file', type=str, help="Path to the input file")
    parser.add_argument('--output_file', type=str, help="Path to the output file")

    args = parser.parse_args()
    convert_file(args.input_file, args.output_file)

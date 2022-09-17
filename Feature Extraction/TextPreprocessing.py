import numpy as np

# Hệ thống dấu câu
PUNC = [",", ".", "?", "(", ")", "+", "-", "'s"]

def readFileIntoLineList(file_name):
    f = open(file_name)
    line_list = []
    for line in f:
        line_list.append(line.strip())
    return line_list

def transferLineToWordList(line_list: list[str]):
    word_list = [] 
    # Tách từ khỏi câu và xóa dấu câu
    for line in line_list:
        temp = line.split(" ")
        for i in temp:
            if len(i) > 2:
                if i[-2:] in PUNC:
                    i = i[:-2]
                if i[0] in PUNC:
                    i = i[1:]
                if i[-1] in PUNC:
                    i = i[:-1]
                i = i.lower()
                word_list.append(i)
    return word_list

def calculateFreq(word_list: list[str]):
    result = {}
    for word in word_list:
        if word in result:
            result[word] += 1
        else:
            result[word] = 1
    return result

def mergeDict(a: list[dict]):
    result = {}
    for freq_dict in a:
        for key in freq_dict.keys():
            if key in result:
                result[key] += freq_dict[key]
            else:
                result[key] = freq_dict[key]
    # Sắp xếp 
    return dict(sorted(result.items(), key=lambda item: item[1], reverse=True))

def createOrderDictionary(general_dict: dict):
    i = 0
    result = {}
    for word in general_dict.keys():
        result[word] = i
        i += 1
    return result

def calculateTFIDF(word, freq_dict, freq_dict_list):
    n = freq_dict[word]
    sum_of_word = 0
    for word in freq_dict.keys():
        sum_of_word += freq_dict[word]
    # Tính tf
    tf = n/sum_of_word
    # Tính idf
    N = len(freq_dict_list)
    count = 0
    for i in freq_dict_list:
        if word in i.keys():
            count += 1
    idf = N / (1 + count)
    # trả về tf-idf
    return tf * idf
    
def createFeatureVector(freq_dict, freq_dict_list, general_dict, order_dict):
    feature_vector = np.zeros(len(general_dict))
    for word in freq_dict:
        position = order_dict[word]
        value = calculateTFIDF(word, freq_dict,freq_dict_list)
        feature_vector[position] = value
    return feature_vector



if __name__ == '__main__':
    freq_dict_list = []
    for i in range(5):
        line_list = readFileIntoLineList(f'vb0{i+1}.txt')
        word_list = transferLineToWordList(line_list)
        dict_freq = calculateFreq(word_list)
        freq_dict_list.append(dict_freq)
    general_dictionary = mergeDict(freq_dict_list)
    order_dictionary = createOrderDictionary(general_dictionary)
    tensor_2D_result = []
    for freq_dict in freq_dict_list:
        feature_vector = createFeatureVector(freq_dict, freq_dict_list, general_dictionary, order_dictionary)
        tensor_2D_result.append(feature_vector)
    tensor_2D_result = np.array(tensor_2D_result)
    print(tensor_2D_result)
    
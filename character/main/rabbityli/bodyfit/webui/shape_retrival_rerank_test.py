from FlagEmbedding import FlagReranker
import numpy as np
import os
import sys
import  json

reranker = FlagReranker('BAAI/bge-reranker-large', use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation



def load_json(j):
    with open(j) as f:
        data = json.load(f)
    return data


# code_file_path = "/aigc_cfs_2/rabbityli/bodyfit/webui/cloth_warpper.py"
code_file_path = "/home/rabbityl/workspace/cloth_wrap/webui/shape_retrival_rerank_test.py"


def shape_text_retrieve(gender, text_input):


    base_body_map = load_json ( os.path.join(os.path.dirname(code_file_path), "base_body_map.json"))
    available_body = load_json( os.path.join(os.path.dirname(code_file_path), "available_body.json"))

    # print( "available_body",available_body)

    avaliable = []
    for k1 in available_body.keys():
        # print("k1", k1)
        for k2 in available_body[k1].keys():
            avaliable.append([k1, k2])

    # print(avaliable)

    body_map_retrival = {"male": {}, "female": {}}


    for e in avaliable:
        g, c = e
        body_map_retrival[g][c] = base_body_map[g][c]

    assert gender in ["male", "female"]

    caption_lst = []
    key_lst = []
    for game, _ in body_map_retrival[gender].items():
        key_lst.append([gender, game])
        caption_lst.append(body_map_retrival[gender][game]["body_caption"])


    query_pair = [ [text_input, s] for s in caption_lst ]
    scores = reranker.compute_score( query_pair )

    # import pdb;
    # pdb.set_trace()
    # for i in range ( len(query_pair)) :
    #     print( i, "-th", "{0:0.2f}".format( scores[i] ), "\t---", key_lst[i] ,"\t---", caption_lst[i])
    index = np.argmax( np.array( scores ) )
    # print( "index:", index)
    return key_lst [ index ]


if __name__ == '__main__':

    import time
    s = time.time()
    # try:
    te = ["小孩儿身材", "成年人身材", "肥胖身材", "强壮身材", "匀称身材", "普通身材",  "瘦弱身材", "圆滚滚身材", "修长身材"]
    # te =["[Child's body]", '[Adult body]', '[Obese Body]', '[Strong Build]', '[Well-proportioned figure]',
    #  '[Average build]', '[slender figure]', '[round body]', '[thin slim]']

    # te = ["plump"]
    for i in range (len(te)):
        print(i, te[i] ,shape_text_retrieve( "female", te[i] ), shape_text_retrieve( "male", te[i] ) )
    # except e :
    #     pass
    print( time.time() - s )
# NPC layer retrieve 

it is for Six kinds of category for 'hair', 'top' and 'bottom', 'shoe', 'outfit', 'others'

## 1. environment: 

  it could refer to requirements.txt

## 2. Calculate CLIP image embeddings by running

  `CLIP_img_gen/calc_clip_img_embedding.py`

## 3. Prepare the packed embedding in one single file

  `CLIP_img_gen/prepare_embeddings.py`

## 4. Generate faiss gpu index

  `CLIP_img_gen/gen_faiss_index_gpu.py`

## 5. Test retrive from objaverse

  `CLIP_img_gen/retrieval.py`

## 6. Deploy the webui
  
  `python webui/gradio_app_image_word_v2.py`




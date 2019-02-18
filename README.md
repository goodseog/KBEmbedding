# KGEmbedding
Knowledge graph Embedding PoC with Paper ; Knowledge Graph Embedding: A Survey of Approaches and Applications(https://ieeexplore.ieee.org/document/8047276)



## TransE

## TransH

## visualize (t-SNE)
Using t-SNE to Display high dimension data on 2D Plane.

``` text
// comment : entity2vec.txt | relation2vec.txt data form
[entity|relation ID #1]    [k-dimension vector_1]
[entity|relation ID #2]    [k-dimension vector_2]
[entity|relation ID #3]    [k-dimension vector_3]
...
```

```bash
# visualize.py run code 
python3 visualize.py [entity_vec.txt|relation_vec.txt]
```


## benchmarks
All datas from
https://github.com/thunlp/OpenKE

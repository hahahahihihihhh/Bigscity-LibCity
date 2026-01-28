### kg2att_rel.py

脚本用于将 poi_kg & adj_kg 转化为KR-EAR知识图谱表征的输入

  python scripts/kg2att_rel.py --dataset NYCTAXI20140103

- train-rel.txt: training file of relations, format (e1, e2, rel).
- ~~test-rel.txt: test file of relations, same format as train-rel.txt.~~
- train-attr.txt: training file of attributes, format (e1, val, attar).
- ~~test-attr.txt: test file of attributes, same format as train-attr.txt.~~
- entity2id.txt: all entities and corresponding ids, one per line.
- relation2id.txt: all relations and corresponding ids, one per line.
- attribute2id.txt: all attributes and corresponding ids, one per line.
- val2id.txt: : all values and corresponding ids, one per line.
- attribute_val.txt: the value set of each attribute

### embedding_txt2assistCsv.py

将KR-EAR模型txt形式的嵌入表示，转化为辅助csv矩阵，并转移至目标位置

### multihop_cormatrix_gen.py

根据KR-EAR学得的嵌入表示，生成KMHNet的多跳关联矩阵并转移至目标位置

### dyna2matrix.py

将dyna形式的交通出入流数据改为matrix形式

### aug_kg_gen.py

用于构建增强知识图谱

### multihop_cor-matrix_gen.py
用于生成KMHNet的多跳矩阵
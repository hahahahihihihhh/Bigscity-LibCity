csv文件名称的含义
    [A]_[B]_[C]d_[D]hop_s[E].csv
        A=poi，代表考虑外部poi关联
        B=S，代表仅考虑邻接关系；B=wSTF，代表考虑带权语义关联图。
        C=64/128/256，代表维度D
        D=1/2/3，代表语义阶数K
        E=1/2/3，代表单图稀疏度

poi_wSTF
    完整模型
wSTF
    无外部poi关联
poi_encode # todo 待完善
    无内部关联
poi_S_encode
    无内部语义关联
encode
    无多源关联感知
S_encode # todo 待完善
    KMHNet
import copy
import os, re
import pickle
import random

import numpy as np
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
from datetime import datetime, timedelta, time


def obatin_spatial_pickle(config, logger):
    spat_pickle_file = config['spat_pickle_file'].format(config.get('dataset'))
    with open(spat_pickle_file, 'rb') as f_pickle:
        dictKG_spatial = pickle.load(f_pickle)
    logger.info('\n******TIME load spatial kg pickle: {}******'.format(datetime.now()))
    return dictKG_spatial


def obatin_temporal_pickle(config, logger):
    temp_pickle_file = config['temp_pickle_file'].format(config.get('dataset'))
    with open(temp_pickle_file, 'rb') as f_pickle:
        dictKG_temporal = pickle.load(f_pickle)  # dict_dynamic {time: {id: keys: {value}}}
    logger.info('\n******TIME load temporal kg pickle: {}******'.format(datetime.now()))
    return dictKG_temporal


def add_factdict_list(dict_fact):
    list_fact = []
    for _key in dict_fact:
        list_fact.extend(dict_fact[_key])
    return len(list_fact), list_fact


def add_factdict_list_bufferfilter(dict_fact, buffer_start, buffer_end):
    list_fact = []
    for _key in dict_fact:
        for _key_value in dict_fact[_key]:
            _fact_rel = _key_value[1].split('Buffer')[1]
            _buffer = int(_fact_rel[1:-1])
            if buffer_start <= _buffer <= buffer_end:
                list_fact.append(_key_value)
    return len(list_fact), list_fact


def generate_spatial_kg(config, logger):
    spat_pickle_file = config['spat_pickle_file'].format(config.get('dataset'))
    spat_model_used = config.get('spat_model_used')
    spat_cont_used = config.get('spat_cont_used')
    spat_link_used = config.get('spat_link_used')
    spat_buffer_used = config.get('spat_buffer_used')
    spat_buffer_list = spat_buffer_used.split('-')
    spat_buffer_start = int(spat_buffer_list[0])
    spat_buffer_end = int(spat_buffer_list[1])
    logger.info('******TIME[Spatial KG] begin: {}******'.format(datetime.now()))
    logger.info('model:{}, spat_cont_used:{}, spat_link_used:{}, buffer:{}-{}'.format(spat_model_used, spat_cont_used, spat_link_used, spat_buffer_start, spat_buffer_end))

    # load and obtain kg spatial data
    logger.info('\n******TIME load spatial kg data: {}******'.format(datetime.now()))
    with open(spat_pickle_file, 'rb') as f_pickle:
        dictKG_spatial = pickle.load(f_pickle)
    fact_used_list = []
    if spat_cont_used != 'none':
        spat_cont_used_list = spat_cont_used.split('-')
        if 'area' in spat_cont_used_list:
            add_list_len, add_list = add_factdict_list(dictKG_spatial['area_spat'])
            fact_used_list.extend(add_list)
            logger.info('    ADD FACT: area_spat with {} items'.format(add_list_len))
        if 'poi' in spat_cont_used_list:
            add_list_len, add_list = add_factdict_list_bufferfilter(dictKG_spatial['poi_spat'], spat_buffer_start, spat_buffer_end)
            fact_used_list.extend(add_list)
            logger.info('    ADD FACT: poi_spat with {} items in buffer[{}-{}]'.format(add_list_len, spat_buffer_start, spat_buffer_end))
        if 'land' in spat_cont_used_list:
            add_list_len, add_list = add_factdict_list_bufferfilter(dictKG_spatial['land_spat'], spat_buffer_start, spat_buffer_end)
            fact_used_list.extend(add_list)
            logger.info('    ADD FACT: land_spat with {} items in buffer[{}-{}]'.format(add_list_len, spat_buffer_start, spat_buffer_end))
    fact_used_len1 = len(fact_used_list)
    logger.info('ADD Spatial FACT with {} items'.format(fact_used_len1))
    if spat_link_used > 0:
        dict_linkbool = dictKG_spatial['link_bool']
        for i_degree in range(1, spat_link_used + 1):
            link_key = 'degree[{}]'.format(str(i_degree))
            fact_used_list.extend(dict_linkbool[link_key])
            logger.info('    ADD FACT: link_bool with {} items in {}'.format(len(dict_linkbool[link_key]), link_key))
    fact_used_len2 = len(fact_used_list) - fact_used_len1
    logger.info('ADD Spatial link FACT with {} items in degree[0-{}]'.format(fact_used_len2, spat_link_used))
    logger.info('ADD [{}] FACTS in total to this experiment'.format(len(fact_used_list)))

    rel_replaced = 'TouchedByArea'
    rel_replaced_num = 0
    for _i_fact in range(len(fact_used_list)):
        _fact_temp = fact_used_list[_i_fact]
        _fact_temp_rel = _fact_temp[1]
        if rel_replaced in _fact_temp_rel:
            rel_replaced_num += 1
            _new_fact = [_fact_temp[0], rel_replaced, _fact_temp[2]]
            fact_used_list[_i_fact] = _new_fact
    print('Replace Spatial rel[TouchedByArea] with {} items'.format(rel_replaced_num))

    # create TriplesFactory
    logger.info('\n******TIME create spatial triples factory: {}******'.format(datetime.now()))
    triples_lines = np.array(fact_used_list)
    logger.info('triples_lines shape: {}'.format(triples_lines.shape))
    tf_spat = TriplesFactory.from_labeled_triples(triples_lines)
    return tf_spat


def add_factdict_list_sub(dict_kgsub, dictKG_spatial, str_flag, logger):
    # basic pars
    dict_fact = dictKG_spatial[str_flag + '_spat']
    dict_attr = dictKG_spatial[str_flag + '_attr']
    area_list = list(dict_fact.keys())
    logger.info('[SUBKG]Areas in KG-SUB starts from [{}] end at [{}]'.format(area_list[0], area_list[-1]))

    # norm ffspeed for dict_attr
    max_value, min_value = -float('inf'), float('inf')
    for _area in area_list:
        attr_list = dict_attr[_area]
        for _attr in attr_list:
            _ffspeed = float(_attr[2])
            if _ffspeed > max_value:
                max_value = _ffspeed
            if _ffspeed < min_value:
                min_value = _ffspeed

    # process
    total_num_fact = 0
    total_num_attr = 0
    for _area in area_list:
        dict_kgsub[_area] = {}
        dict_kgsub[_area][str_flag] = []
        fact_list = dict_fact[_area]
        attr_list = dict_attr[_area]
        _attr_value_self = 0
        for _fact in fact_list:
            total_num_fact += 1
            _attr_value, _attr_value_flag = 0, 0  # flag
            # find the corresponding fact and attr for _fact
            _rel_fact_id = re.findall(r'\[(.*?)\]', _fact[1])
            if len(_rel_fact_id) != 1:
                logger.info('[ERROR]There are [{}]!=1 in fact:{}'.format(len(_rel_fact_id), _fact[1]))
                exit()
            # search attr
            for _attr in attr_list:
                _rel_attr_id = re.findall(r'\[(.*?)\]', _attr[1])
                if len(_rel_attr_id) != 1:
                    logger.info('[ERROR]There are [{}]!=1 in attr:{}'.format(len(_rel_attr_id), _attr[1]))
                    exit()
                if _fact[0] == _attr[0] and _rel_fact_id[0] == _rel_attr_id[0]:
                    # _attr_value = (float(_attr[2]) - min_value) / (max_value - min_value)  # norm
                    _attr_value = 1
                    _attr_value_flag += 1
                if _fact[0] == _attr[0] and _rel_attr_id[0] == '0':
                    # _attr_value_self = (float(_attr[2]) - min_value) / (max_value - min_value)  # norm
                    _attr_value_self = 1
            # finish search
            if _attr_value_flag == 1:  # flag
                _fact_cp = copy.deepcopy(_fact)
                _fact_cp.append(_attr_value)
                _fact_cp[1] = 'TouchedByArea'  # replace 'TouchedByArea[1-N]' by 'TouchedByArea'
                dict_kgsub[_area][str_flag].append(_fact_cp)
                total_num_attr += 1
            else:
                logger.info('[ERROR]Find [{}] attr-value for fact:{}'.format(_attr_value_flag, _fact))
                exit()
        # for itself
        _self_fact = [_area, 'self_ffspeed', _area, _attr_value_self]  # 0, freeflowspeed itself
        dict_kgsub[_area][str_flag].append(_self_fact)
    if total_num_fact != total_num_attr:
        logger.info('[ERROR]Fact number:[{}] not equal to Attr number:[{}]'.format(total_num_fact, total_num_attr))
        exit()
    return total_num_fact


def add_factdict_list_bufferfilter_sub(dict_kgsub, dictKG_spatial, str_flag, buffer_str, logger):
    # basic pars
    dict_fact = dictKG_spatial[str_flag + '_spat']
    dict_attr = dictKG_spatial[str_flag + '_attr']
    area_list = list(dict_fact.keys())
    buffer_list = buffer_str.split('-')
    buffer_start = int(buffer_list[0])
    buffer_end = int(buffer_list[1])

    # norm
    max_value, min_value = -float('inf'), float('inf')
    for _area in area_list:
        attr_list = dict_attr[_area]
        for _attr in attr_list:
            _attr_value = re.findall(r'\[(.*?)\]', _attr[2])
            if len(_attr_value) != 1:
                logger.info('[ERROR]There are [{}]!=1 in attr:{}'.format(len(_attr_value), _attr[2]))
                exit()
            _attr_value = float(_attr_value[0])
            # obtain max and min value
            if float(_attr_value) > max_value:
                max_value = float(_attr_value)
            if float(_attr_value) < min_value:
                min_value = float(_attr_value)

    # process
    total_num_fact = 0
    total_num_attr = 0
    for _area in area_list:
        if _area not in dict_kgsub:
            dict_kgsub[_area] = {}
        dict_kgsub[_area][str_flag] = []
        fact_list = dict_fact[_area]
        attr_list = dict_attr[_area]
        for _fact in fact_list:
            _attr_value, _attr_value_flag = 0, 0  # flag
            # find the corresponding fact and attr for _fact
            _rel_fact_id = re.findall(r'\[(.*?)\]', _fact[1])
            if len(_rel_fact_id) != 1:
                logger.info('[ERROR]There are [{}]!=1 in fact:{}'.format(len(_rel_fact_id), _fact[1]))
                exit()
            _buffer = int(_rel_fact_id[0])
            if _buffer < buffer_start or _buffer > buffer_end:  # only consider facts within buffer
                continue
            total_num_fact += 1
            # search attr
            for _attr in attr_list:
                _rel_attr_id = re.findall(r'\[(.*?)\]', _attr[1])
                if len(_rel_attr_id) != 2:
                    logger.info('[ERROR]There are [{}]!=2 in attr:{}'.format(len(_rel_attr_id), _attr[1]))
                    exit()
                if _fact[0] == _rel_attr_id[0] and _rel_fact_id[0] == _rel_attr_id[1] and _fact[2] == _attr[0]:
                    # Type=[Type] and [DIST]=[DIST] and areaID=areaID
                    _attr_value = re.findall(r'\[(.*?)\]', _attr[2])
                    if len(_attr_value) != 1:
                        logger.info('[ERROR]There are [{}]!=1 in attr:{}'.format(len(_attr_value), _attr[2]))
                        exit()
                    _attr_value = (float(_attr_value[0]) - min_value) / (max_value - min_value)
                    _attr_value_flag += 1
            # finish search
            if _attr_value_flag == 1:  # flag
                _fact_cp = copy.deepcopy(_fact)
                _fact_cp.append(_attr_value)
                dict_kgsub[_area][str_flag].append(_fact_cp)
                total_num_attr += 1
            else:
                logger.info('[ERROR]Find [{}] attr-value for fact:{}'.format(_attr_value_flag, _fact))
                exit()
    if total_num_fact != total_num_attr:
        logger.info('[ERROR]Fact number:[{}] not equal to Attr number:[{}]'.format(total_num_fact, total_num_attr))
        exit()
    return total_num_fact


def generate_kgsub_spat(config, logger, dictKG_spatial):
    # the final dict_kgsub
    dict_kgsub = {}

    # basic par
    spat_attr_used = config.get('spat_attr_used')
    spat_link_attr = config.get('spat_link_attr')
    spat_buffer_attr = config.get('spat_buffer_attr')
    logger.info('******TIME[Spatial SUBKG] begin: {}******'.format(datetime.now()))
    logger.info('[SUBKG]spat_attr_used:{}, link:{}, buffer:{}'.format(spat_attr_used, spat_link_attr, spat_buffer_attr))

    # generate subkg
    if spat_attr_used != 'none':
        spat_attr_used_list = spat_attr_used.split('-')
        if 'area' in spat_attr_used_list:
            fact_num1 = add_factdict_list_sub(dict_kgsub, dictKG_spatial, 'area', logger)
            logger.info('[SUBKG]ADD FACT/ATTR: area with {} items'.format(fact_num1))
        if 'poi' in spat_attr_used_list:
            fact_num2 = add_factdict_list_bufferfilter_sub(dict_kgsub, dictKG_spatial, 'poi', spat_buffer_attr, logger)
            logger.info('[SUBKG]ADD FACT/ATTR: poi with {} items'.format(fact_num2))
        if 'land' in spat_attr_used_list:
            fact_num3 = add_factdict_list_bufferfilter_sub(dict_kgsub, dictKG_spatial, 'land', spat_buffer_attr, logger)
            logger.info('[SUBKG]ADD FACT/ATTR: land with {} items'.format(fact_num3))

    if spat_link_attr > 0:
        fact_num4 = 0
        area_list = list(dictKG_spatial['area_spat'].keys())
        dict_linknum = dictKG_spatial['link_num']
        # norm
        max_value, min_value = -float('inf'), float('inf')
        for _area in area_list:
            for i_degree in range(1, spat_link_attr + 1):
                link_key = 'degree[{}]'.format(str(i_degree))
                _attr_list = dict_linknum[link_key]
                for _attr in _attr_list:
                    _link_num = re.findall(r'\d+', _attr[3])  # num of links
                    if len(_link_num) != 1:
                        logger.info('[ERROR]There are num:[{}]!=1 in fact:{}'.format(len(_link_num), _attr))
                        exit()
                    if int(_link_num[0]) > max_value:
                        max_value = int(_link_num[0])
                    if int(_link_num[0]) < min_value:
                        min_value = int(_link_num[0])
        # process
        for _area in area_list:
            if _area not in dict_kgsub:
                dict_kgsub[_area] = {}
            dict_kgsub[_area]['link'] = []
            for i_degree in range(1, spat_link_attr + 1):
                link_key = 'degree[{}]'.format(str(i_degree))
                _attr_list = dict_linknum[link_key]
                for _attr in _attr_list:
                    if _area != _attr[0]:
                        continue
                    _link_num = re.findall(r'\d+', _attr[3])  # num of links
                    if len(_link_num) != 1:
                        logger.info('[ERROR]There are num:[{}]!=1 in fact:{}'.format(len(_link_num), _attr))
                        exit()
                    fact_num4 += 1
                    _attr_cp = copy.deepcopy(_attr)
                    epsilon = 0.000001
                    if abs(max_value - min_value) > epsilon:
                        _attr_cp[3] = float(int(_link_num[0]) - min_value) / float(max_value - min_value)  # replace num:[] by an int
                    else:
                        _attr_cp[3] = 0.0
                    dict_kgsub[_area]['link'].append(_attr_cp)
        logger.info('[SUBKG]ADD FACT/ATTR: link with {} items'.format(fact_num4))
    logger.info('******TIME[Spatial SUBKG] end: {}******'.format(datetime.now()))
    return dict_kgsub  # {area_id: {area/poi/land/link: [fact]}}


def add_tempfact_time(dictKG_temporal, datetime_list):
    list_fact = []
    for _datetime in datetime_list:
        dictKG_id = dictKG_temporal[_datetime]
        for _id in dictKG_id:
            dictKG_keys = dictKG_id[_id]
            area_id = 'area_' + str(_id)
            list_fact.append([area_id, 'hourOfDay', 'Hour'])
            list_fact.append([area_id, 'dayOfWeek', 'Day'])
            list_fact.append([area_id, 'isRestWork', 'DayType'])
    return len(list_fact), list_fact


def add_tempdict_fact(dictKG_temporal, datetime_list, time_used, link_used, name_sur, name_key):
    list_fact = []
    # current and past datetime
    for _datetime in datetime_list:
        # time list: [current, past, ...]
        timelist_used = []
        for i in range(0, time_used):
            _pastmins = 60 * i
            _pasttime = _datetime - timedelta(minutes=_pastmins)
            timelist_used.append(_pasttime)
        # add list
        list_id = list(dictKG_temporal[_datetime].keys())
        for _num in range(0, len(timelist_used)):
            for _id in list_id:  # cal each area_id
                area_id = 'area_' + str(_id)
                if _num == 0:  # current value
                    _currenttime = timelist_used[0]
                    _currentvalue = round(dictKG_temporal[_currenttime][_id][name_key])
                    list_fact.append([area_id, 'has{}Current'.format(name_sur), '{}'.format(name_sur)])
                else:  # AvePast[{}]min
                    flag_time = True
                    ave_value = 0.0
                    for i in range(0, _num+1):
                        _usedtime = timelist_used[i]
                        if _usedtime not in dictKG_temporal:
                            flag_time = False
                            continue
                        ave_value += dictKG_temporal[_usedtime][_id][name_key]
                    if flag_time:
                        ave_value = round(ave_value / (_num+1))
                        list_fact.append([area_id, 'has{}Ave[{}]min'.format(name_sur, str((_num+1)*60)), '{}'.format(name_sur)])
    len_fact_timeused = len(list_fact)

    # linked datetime
    if link_used != 'none':
        for _datetime in datetime_list:
            dictKG_id = dictKG_temporal[_datetime]
            for _id in dictKG_id:
                dictKG_keys = dictKG_id[_id]
                area_id = 'area_' + str(_id)
                # linked value: hour/day/week
                link_used_list = link_used.split('-')
                if 'hour' in link_used_list:
                    list_fact.append([area_id, 'has{}Hourly'.format(name_sur), '{}'.format(name_sur)])
                if 'day' in link_used_list:
                    list_fact.append([area_id, 'has{}Daily'.format(name_sur), '{}'.format(name_sur)])
                if 'week' in link_used_list:
                    list_fact.append([area_id, 'has{}Weekly'.format(name_sur), '{}'.format(name_sur)])
    len_fact_linkused = len(list_fact) - len_fact_timeused
    return len_fact_timeused, len_fact_linkused, list_fact


def generate_temporal_kg(config, logger, kg_logger=True):
    temp_pickle_file = config['temp_pickle_file'].format(config.get('dataset'))
    temp_model_used = config.get('temp_model_used')
    temp_cont_used = config.get('temp_cont_used')
    temp_time_used = config.get('temp_time_used')
    temp_link_used = config.get('temp_link_used')
    temp_datetime = config.get('temp_datetime')
    logger.info('******TIME[Temporal KG] begin: {}******'.format(datetime.now()))
    logger.info('model:{}, temp_cont_used:{}, temp_time_used:{}, temp_link_used:{}, date_time:{}'
                .format(temp_model_used, temp_cont_used, temp_time_used, temp_link_used, temp_datetime))
    datetime_used = datetime.strptime(temp_datetime.strip(), '%Y-%m-%dT%H:%M:%SZ')
    datetime_list = [datetime_used]

    # load and obtain kg temporal data
    if kg_logger: logger.info('\n******TIME load temporal kg data: {}******'.format(datetime.now()))
    with open(temp_pickle_file, 'rb') as f_pickle:
        dictKG_temporal = pickle.load(f_pickle)  # dict_dynamic {time: {id: keys: {value}}}
    fact_used_list = []
    if temp_cont_used != 'none':
        temp_cont_used_list = temp_cont_used.split('-')
        if 'time' in temp_cont_used_list:
            add_list_len, add_list = add_tempfact_time(dictKG_temporal, datetime_list)  # consider each datetime in list
            fact_used_list.extend(add_list)
            if kg_logger: logger.info('    ADD FACT: time(time/hour/day/rest_work) with {} items'.format(add_list_len))
        if 'jam' in temp_cont_used_list:
            len_fact_timeused, len_fact_linkused, add_list = add_tempdict_fact(dictKG_temporal, datetime_list, temp_time_used, temp_link_used, 'jam', 'jf')
            fact_used_list.extend(add_list)
            if kg_logger: logger.info('    ADD FACT: jam with {} timeAVE items and {} timeLINK items'.format(len_fact_timeused, len_fact_linkused))
        if 'weather' in temp_cont_used_list:
            len_fact_timeused, len_fact_linkused, add_list = add_tempdict_fact(dictKG_temporal, datetime_list, temp_time_used, temp_link_used, 'tprt', 'temperature')
            fact_used_list.extend(add_list)
            if kg_logger: logger.info('    ADD FACT: temperature with {} timeAVE items and {} timeLINK items'.format(len_fact_timeused, len_fact_linkused))
            len_fact_timeused, len_fact_linkused, add_list = add_tempdict_fact(dictKG_temporal, datetime_list, temp_time_used, temp_link_used, 'rain', 'rainfall')
            fact_used_list.extend(add_list)
            if kg_logger: logger.info('    ADD FACT: rainfall with {} timeAVE items and {} timeLINK items'.format(len_fact_timeused, len_fact_linkused))
            len_fact_timeused, len_fact_linkused, add_list = add_tempdict_fact(dictKG_temporal, datetime_list, temp_time_used, temp_link_used, 'wind', 'windspeed')
            fact_used_list.extend(add_list)
            if kg_logger: logger.info('    ADD FACT: windspeed with {} timeAVE items and {} timeLINK items'.format(len_fact_timeused, len_fact_linkused))
    logger.info('ADD [{}] FACTS in total to this experiment'.format(len(fact_used_list)))

    # create TriplesFactory
    if kg_logger: logger.info('\n******TIME create temporal triples factory: {}******'.format(datetime.now()))
    triples_lines = np.array(fact_used_list)
    logger.info('triples_lines shape: {}'.format(triples_lines.shape))
    tf_temp = TriplesFactory.from_labeled_triples(triples_lines)
    return tf_temp


def add_tempfact_time_sub(dict_kgsub, dictKG_temporal, temp_datetime, str_flag):
    # dictKG_temporal: {time: {id: keys: {value}}}
    dictKG_id = dictKG_temporal[temp_datetime]
    total_num_attr = 0
    for _id in dictKG_id:
        area_id = 'area_' + str(_id)
        if area_id not in dict_kgsub:
            dict_kgsub[area_id] = {}
        dict_kgsub[area_id][str_flag] = []
        # add attr
        dictKG_keys = dictKG_id[_id]
        _hour_of_day = np.cos(float(dictKG_keys['hour_of_day'] - 1) / 24. * 2. * np.pi)
        dict_kgsub[area_id][str_flag].append([area_id, 'hourOfDay', 'Hour', _hour_of_day])
        _day_of_week = np.cos(float(dictKG_keys['day_of_week'] - 1) / 7. * 2. * np.pi)
        dict_kgsub[area_id][str_flag].append([area_id, 'dayOfWeek', 'Day', _day_of_week])
        _rest_work = 0
        if dictKG_keys['rest_work'] == 'w': _rest_work = 1
        if dictKG_keys['rest_work'] == 'r': _rest_work = -1
        dict_kgsub[area_id][str_flag].append([area_id, 'isRestWork', 'DayType', _rest_work])
        total_num_attr += 3
    return total_num_attr


def add_tempdict_fact_sub(logger, dict_kgsub, dictKG_temporal, temp_datetime, time_used, link_used, name_sur, name_key):
    max_value, min_value = 0, 0
    if name_sur is 'jam':
        max_value, min_value = 10.0, 0.0
    if name_sur is 'tprt':
        max_value, min_value = 19.7, -21.3   # 40.0, 20.0
    if name_sur is 'rain':
        max_value, min_value = 7.6, 0.0    # 20.0, 0.0
    if name_sur is 'wind':
        max_value, min_value = 63.3, 0.0    # 30.0, 0.0

    # current and past datetime
    time_attr_num = 0
    timelist_used = []
    for i in range(0, time_used):
        _pastmins = 60 * i
        _pasttime = temp_datetime - timedelta(minutes=_pastmins)
        timelist_used.append(_pasttime)
    list_id = list(dictKG_temporal[temp_datetime].keys())
    for _num in range(0, len(timelist_used)):
        for _id in list_id:  # cal each area_id
            area_id = 'area_' + str(_id)
            if area_id not in dict_kgsub:
                dict_kgsub[area_id] = {}
            if name_sur not in dict_kgsub[area_id]:
                dict_kgsub[area_id][name_sur] = []
            if _num == 0:  # current value
                _currenttime = timelist_used[0]
                _currentvalue = (float(dictKG_temporal[_currenttime][_id][name_key]) - min_value) / (max_value - min_value)
                _new_fact = [area_id, 'has{}Current'.format(name_sur), '{}'.format(name_sur), _currentvalue]
                dict_kgsub[area_id][name_sur].append(_new_fact)
                time_attr_num += 1
            else:  # AvePast[{}]min
                flag_time = True
                ave_value = 0.0
                for i in range(0, _num+1):
                    _usedtime = timelist_used[i]
                    if _usedtime not in dictKG_temporal:
                        flag_time = False
                        continue
                    ave_value += dictKG_temporal[_usedtime][_id][name_key]
                if flag_time:
                    ave_value = (float(ave_value / float(_num+1)) - min_value) / (max_value - min_value)
                    _new_fact = [area_id, 'has{}Ave[{}]min'.format(name_sur, str((_num+1)*60)), '{}'.format(name_sur), ave_value]
                    dict_kgsub[area_id][name_sur].append(_new_fact)
                    time_attr_num += 1

    # linked datetime
    link_attr_num = 0
    if link_used != 'none':
        dictKG_id = dictKG_temporal[temp_datetime]
        for _id in dictKG_id:
            dictKG_keys = dictKG_id[_id]
            area_id = 'area_' + str(_id)
            if area_id not in dict_kgsub:
                logger.info('[ERROR]No area:{} in link construction'.format(area_id))
            if name_sur not in dict_kgsub[area_id]:
                logger.info('[ERROR]No name:{} in link construction'.format(name_sur))
            # linked value: hour/day/week
            link_used_list = link_used.split('-')
            if 'hour' in link_used_list:
                _value = (float(dictKG_keys[name_key + '_hour']) - min_value) / (max_value - min_value)
                _new_fact = [area_id, 'has{}Hourly'.format(name_sur), '{}'.format(name_sur), _value]
                dict_kgsub[area_id][name_sur].append(_new_fact)
                link_attr_num += 1
            if 'day' in link_used_list:
                _value = (float(dictKG_keys[name_key + '_day']) - min_value) / (max_value - min_value)
                _new_fact = [area_id, 'has{}Daily'.format(name_sur), '{}'.format(name_sur), _value]
                dict_kgsub[area_id][name_sur].append(_new_fact)
                link_attr_num += 1
            if 'week' in link_used_list:
                try:
                    _value = (float(dictKG_keys[name_key + '_week']) - min_value) / (max_value - min_value)
                except:
                    _value = 0
                    logger.info('[NODATA]dictKG_keys[{} + _week] in {} doesnot exist'.format(name_key, temp_datetime))
                _new_fact = [area_id, 'has{}Weekly'.format(name_sur), '{}'.format(name_sur), _value]
                dict_kgsub[area_id][name_sur].append(_new_fact)
                link_attr_num += 1
    return time_attr_num, link_attr_num


def add_tempdict_fact_sub_notcover(logger, dict_kgsub, dictKG_temporal, temp_datetime, time_used, link_used, name_sur, name_key, temp_datetime_list, kg_weight_temp):
    max_value, min_value = 0, 0
    if name_sur is 'jam':
        max_value, min_value = 10.0, 0.0
    if name_sur is 'tprt':
        max_value, min_value = 19.7, -21.3   # 40.0, 20.0
    if name_sur is 'rain':
        max_value, min_value = 7.6, 0.0    # 20.0, 0.0
    if name_sur is 'wind':
        max_value, min_value = 63.3, 0.0    # 30.0, 0.0

    # current and past datetime
    time_attr_num = 0
    timelist_used = []
    for i in range(0, time_used):
        _pastmins = 60 * i
        _pasttime = temp_datetime - timedelta(minutes=_pastmins)
        timelist_used.append(_pasttime)
    list_id = list(dictKG_temporal[temp_datetime].keys())
    for _num in range(0, len(timelist_used)):
        for _id in list_id:  # cal each area_id
            area_id = 'area_' + str(_id)
            if area_id not in dict_kgsub:
                dict_kgsub[area_id] = {}
            if name_sur not in dict_kgsub[area_id]:
                dict_kgsub[area_id][name_sur] = []
            if _num == 0:  # current value
                _currenttime = timelist_used[0]
                _currentvalue = (float(dictKG_temporal[_currenttime][_id][name_key]) - min_value) / (max_value - min_value)
                _new_fact = [area_id, 'has{}Current'.format(name_sur), '{}'.format(name_sur), _currentvalue]
                dict_kgsub[area_id][name_sur].append(_new_fact)
                time_attr_num += 1
            else:  # AvePast[{}]min
                flag_time = True
                ave_value = 0.0
                ave_number = 0
                for i in range(0, _num+1):
                    _usedtime = timelist_used[i]
                    if _usedtime not in dictKG_temporal:
                        flag_time = False
                        continue
                    if _usedtime in temp_datetime_list:  # not_cover: don't include context data not in the time
                        ave_value += dictKG_temporal[_usedtime][_id][name_key]
                        ave_number += 1
                if flag_time:
                    ave_value = (float(ave_value / float(ave_number)) - min_value) / (max_value - min_value)
                    if ave_number == 0 and kg_weight_temp == 'times':  # not_cover: for times, set it as 1.0 since we need embeddings's info
                        ave_value = 1.0
                    if ave_number == 0 and kg_weight_temp == 'add':  # not_cover: for add, set it as 0.0 since we need embeddings's info
                        ave_value = 0.0
                    _new_fact = [area_id, 'has{}Ave[{}]min'.format(name_sur, str((_num+1)*60)), '{}'.format(name_sur), ave_value]
                    dict_kgsub[area_id][name_sur].append(_new_fact)
                    time_attr_num += 1

    # linked datetime
    link_attr_num = 0
    if link_used != 'none':
        dictKG_id = dictKG_temporal[temp_datetime]
        for _id in dictKG_id:
            dictKG_keys = dictKG_id[_id]
            area_id = 'area_' + str(_id)
            if area_id not in dict_kgsub:
                logger.info('[ERROR]No area:{} in link construction'.format(area_id))
            if name_sur not in dict_kgsub[area_id]:
                logger.info('[ERROR]No name:{} in link construction'.format(name_sur))
            # linked value: hour/day/week
            link_used_list = link_used.split('-')
            if 'hour' in link_used_list:
                try:
                    _value = (float(dictKG_keys[name_key + '_hour']) - min_value) / (max_value - min_value)
                except:
                    _value = 0.0
                    logger.info('[NODATA]dictKG_keys[{} + _hour] in {} doesnot exist'.format(name_key, temp_datetime))
                # not cover: hour
                _hourtime = temp_datetime - timedelta(hours=1)
                if _hourtime not in temp_datetime_list:
                    if kg_weight_temp == 'add':
                        _value = 0.0
                    elif kg_weight_temp == 'times':
                        _value = 1.0
                _new_fact = [area_id, 'has{}Hourly'.format(name_sur), '{}'.format(name_sur), _value]
                dict_kgsub[area_id][name_sur].append(_new_fact)
                link_attr_num += 1
            if 'day' in link_used_list:
                try:
                    _value = (float(dictKG_keys[name_key + '_day']) - min_value) / (max_value - min_value)
                except:
                    _value = 0.0
                    logger.info('[NODATA]dictKG_keys[{} + _day] in {} doesnot exist'.format(name_key, temp_datetime))
                # not cover: day
                minus_day = 1
                if temp_datetime.weekday() == 0:  # Monday, its last workday is Friday
                    minus_day = 3
                _daytime = temp_datetime - timedelta(days=minus_day)
                if _daytime not in temp_datetime_list:
                    if kg_weight_temp == 'add':
                        _value = 0.0
                    elif kg_weight_temp == 'times':
                        _value = 1.0
                _new_fact = [area_id, 'has{}Daily'.format(name_sur), '{}'.format(name_sur), _value]
                dict_kgsub[area_id][name_sur].append(_new_fact)
                link_attr_num += 1
            if 'week' in link_used_list:
                try:
                    _value = (float(dictKG_keys[name_key + '_week']) - min_value) / (max_value - min_value)
                except:
                    _value = 0.0
                    logger.info('[NODATA]dictKG_keys[{} + _week] in {} doesnot exist'.format(name_key, temp_datetime))
                # not cover: week
                _weektime = temp_datetime - timedelta(days=7)
                if _weektime not in temp_datetime_list:
                    if kg_weight_temp == 'add':
                        _value = 0.0
                    elif kg_weight_temp == 'times':
                        _value = 1.0
                _new_fact = [area_id, 'has{}Weekly'.format(name_sur), '{}'.format(name_sur), _value]
                dict_kgsub[area_id][name_sur].append(_new_fact)
                link_attr_num += 1
    return time_attr_num, link_attr_num


def generate_kgsub_temp(config, logger, dictKG_temporal, temp_datetime, kg_logger=False):
    # dict_kgsub; dictKG_temporal: {time: {id: keys: {value}}}
    dict_kgsub = {}

    # basic par
    temp_attr_used = config.get('temp_attr_used')
    temp_time_attr = config.get('temp_time_attr')
    temp_link_attr = config.get('temp_link_attr')
    if kg_logger: logger.info('******TIME[Temporal SUBKG] begin: {}******'.format(datetime.now()))
    if kg_logger: logger.info('[SUBKG]temp_attr_used:{}, temp_time_attr:{}, temp_link_attr:{}, date_time:{}'
                              .format(temp_attr_used, temp_time_attr, temp_link_attr, temp_datetime))
    if type(temp_datetime) is str:
        temp_datetime = datetime.strptime(temp_datetime.strip(), '%Y-%m-%dT%H:%M:%SZ')

    # generate subkg
    if temp_attr_used != 'none':
        temp_attr_used_list = temp_attr_used.split('-')
        if 'time' in temp_attr_used_list:
            attr_num1 = add_tempfact_time_sub(dict_kgsub, dictKG_temporal, temp_datetime, 'time')
            if kg_logger: logger.info('[SUBKG]ADD FACT: time with {} items'.format(attr_num1))
        if 'jam' in temp_attr_used_list:
            time_num2, link_num2 = add_tempdict_fact_sub(logger, dict_kgsub, dictKG_temporal, temp_datetime, temp_time_attr, temp_link_attr, 'jam', 'jf')
            if kg_logger: logger.info('[SUBKG]ADD FACT: jam with {} timeAVE items and {} timeLINK items'.format(time_num2, link_num2))
        if 'weather' in temp_attr_used_list:
            time_num3, link_num3 = add_tempdict_fact_sub(logger, dict_kgsub, dictKG_temporal, temp_datetime, temp_time_attr, temp_link_attr, 'tprt', 'temperature')
            if kg_logger: logger.info('[SUBKG]ADD FACT: temperature with {} timeAVE items and {} timeLINK items'.format(time_num3, link_num3))
            time_num4, link_num4 = add_tempdict_fact_sub(logger, dict_kgsub, dictKG_temporal, temp_datetime, temp_time_attr, temp_link_attr, 'rain', 'rainfall')
            if kg_logger: logger.info('[SUBKG]ADD FACT: rainfall with {} timeAVE items and {} timeLINK items'.format(time_num4, link_num4))
            time_num5, link_num5 = add_tempdict_fact_sub(logger, dict_kgsub, dictKG_temporal, temp_datetime, temp_time_attr, temp_link_attr, 'wind', 'windspeed')
            if kg_logger: logger.info('[SUBKG]ADD FACT: windspeed with {} timeAVE items and {} timeLINK items'.format(time_num5, link_num5))
    if kg_logger: logger.info('******TIME[Temporal SUBKG] end: {}******'.format(datetime.now()))
    return dict_kgsub


def generate_kgsub_temp_notcover(config, logger, dictKG_temporal, temp_datetime, temp_datetime_list, kg_weight_temp, kg_logger=False):
    # dict_kgsub; dictKG_temporal: {time: {id: keys: {value}}}
    dict_kgsub = {}

    # basic par
    temp_attr_used = config.get('temp_attr_used')
    temp_time_attr = config.get('temp_time_attr')
    temp_link_attr = config.get('temp_link_attr')
    if kg_logger: logger.info('******TIME[Temporal SUBKG] begin: {}******'.format(datetime.now()))
    if kg_logger: logger.info('[SUBKG]temp_attr_used:{}, temp_time_attr:{}, temp_link_attr:{}, date_time:{}'
                              .format(temp_attr_used, temp_time_attr, temp_link_attr, temp_datetime))
    if type(temp_datetime) is str:
        temp_datetime = datetime.strptime(temp_datetime.strip(), '%Y-%m-%dT%H:%M:%SZ')
    # generate subkg
    if temp_attr_used != 'none':
        temp_attr_used_list = temp_attr_used.split('-')
        if 'time' in temp_attr_used_list:
            attr_num1 = add_tempfact_time_sub(dict_kgsub, dictKG_temporal, temp_datetime, 'time')
            if kg_logger: logger.info('[SUBKG]ADD FACT: time with {} items'.format(attr_num1))
        if 'jam' in temp_attr_used_list:
            time_num2, link_num2 = add_tempdict_fact_sub_notcover(logger, dict_kgsub, dictKG_temporal, temp_datetime, temp_time_attr, temp_link_attr, 'jam', 'jf', temp_datetime_list, kg_weight_temp)
            if kg_logger: logger.info('[SUBKG]ADD FACT: jam with {} timeAVE items and {} timeLINK items'.format(time_num2, link_num2))
        if 'weather' in temp_attr_used_list:
            time_num3, link_num3 = add_tempdict_fact_sub_notcover(logger, dict_kgsub, dictKG_temporal, temp_datetime, temp_time_attr, temp_link_attr, 'tprt', 'temperature', temp_datetime_list, kg_weight_temp)
            if kg_logger: logger.info('[SUBKG]ADD FACT: temperature with {} timeAVE items and {} timeLINK items'.format(time_num3, link_num3))
            time_num4, link_num4 = add_tempdict_fact_sub_notcover(logger, dict_kgsub, dictKG_temporal, temp_datetime, temp_time_attr, temp_link_attr, 'rain', 'rainfall', temp_datetime_list, kg_weight_temp)
            if kg_logger: logger.info('[SUBKG]ADD FACT: rainfall with {} timeAVE items and {} timeLINK items'.format(time_num4, link_num4))
            time_num5, link_num5 = add_tempdict_fact_sub_notcover(logger, dict_kgsub, dictKG_temporal, temp_datetime, temp_time_attr, temp_link_attr, 'wind', 'windspeed', temp_datetime_list, kg_weight_temp)
            if kg_logger: logger.info('[SUBKG]ADD FACT: windspeed with {} timeAVE items and {} timeLINK items'.format(time_num5, link_num5))
    if kg_logger: logger.info('******TIME[Temporal SUBKG] end: {}******'.format(datetime.now()))
    return dict_kgsub


def kg_entity_id_from_name(tf_facts, ctx_name):
    entity_list = []
    entityid_list = []
    entity_label = tf_facts.entity_labeling.label_to_id
    for _label in entity_label:
        if ctx_name in _label:
            entity_list.append(_label)
            entityid_list.append(entity_label[_label])
    return entity_list, entityid_list


def kg_entity_id_from_list(tf_facts, lab_list):
    entityid_list = []
    entity_label = tf_facts.entity_labeling.label_to_id
    for _label in lab_list:
        entityid_list.append(entity_label[_label])
    return entityid_list


def kg_embedding(tf_facts, config, logger, model_used, kg_logger=False):
    rand_seed = config.get('seed')
    embed_dim = config.get('kg_embed_dim')
    epochs_num = config.get('kg_epochs_num')

    # split triples factory
    training, testing = tf_facts.split(ratios=0.9, random_state=rand_seed)
    if kg_logger: logger.info('TriplesFactory all  :', tf_facts)
    if kg_logger: logger.info('TriplesFactory train:', training)
    if kg_logger: logger.info('TriplesFactory test :', testing)

    logger.info('\n******TIME[KG modelling] start training {}: {}******'.format(model_used, datetime.now()))
    if model_used == 'RGCN':
        pipeline_result = pipeline(training=training, testing=testing, model=model_used, random_seed=rand_seed,
                                   epochs=epochs_num, training_kwargs=dict(use_tqdm_batch=False, sampler="schlichtkrull"),
                                   evaluation_kwargs=dict(use_tqdm=False))
    else:
        pipeline_result = pipeline(training=training, testing=testing, model=model_used, random_seed=rand_seed,
                                   epochs=epochs_num, training_kwargs=dict(use_tqdm_batch=False), evaluation_kwargs=dict(use_tqdm=False),
                                   model_kwargs=dict(embedding_dim=embed_dim))
    logger.info('******TIME[KG modelling] end training {}: {}******\n'.format(model_used, datetime.now()))
    model = pipeline_result.model
    if kg_logger: logger.info('pipeline_result: ', pipeline_result)

    entity_embedding_tensor = model.entity_representations[0](indices=None)  # [entity_num, embed_dim]
    relation_embedding_tensor = model.relation_representations[0](indices=None)  # [relation_num, embed_dim]
    entity_embedding_numpy = model.entity_representations[0](indices=None).cpu().detach().numpy()
    relation_embedding_numpy = model.relation_representations[0](indices=None).cpu().detach().numpy()
    return entity_embedding_numpy, relation_embedding_numpy


def cal_ent_via_rel(config, logger, model, ent, rel, weight):
    ent_change = []
    if config['kg_weight'] == 'add':
        if model == 'ComplEx':
            ent_change = np.dot(np.diag(rel), ent) + weight
            ent_change = np.real(ent_change)
        elif model == 'KG2E':
            ent_change = ent - rel + weight
        elif model == 'AutoSF':
            ent_change = np.dot(np.diag(rel), ent) + weight
        else:
            logger.info('[ERROR]model-{} was not used in embeddings'.format(model))
    elif config['kg_weight'] == 'times':
        if model == 'ComplEx':
            ent_change = weight * np.dot(np.diag(rel), ent)
            ent_change = np.real(ent_change)
        elif model == 'KG2E':
            ent_change = weight * (ent - rel)
        elif model == 'AutoSF':
            ent_change = weight * np.dot(np.diag(rel), ent)
        else:
            logger.info('[ERROR]model-{} was not used in embeddings'.format(model))
    return np.array(ent_change)


def obtain_temp_kge_final(config, logger, x_goal, x_auxi, dict_kge, temp_datetime):
    # basic par
    temp_attr_used = config.get('temp_attr_used')
    temp_model_used = config.get('temp_model_used')
    embed_dim = config.get('kg_embed_dim')
    kg_context = config.get('kg_context')
    temp_ent_kge, temp_rel_kge = dict_kge['temp_ent_kge'], dict_kge['temp_rel_kge']
    temp_ent_label, temp_rel_label = dict_kge['temp_ent_label'], dict_kge['temp_rel_label']

    # sub_kg for temp
    subdict_temp = dict_kge['sub_temp'][temp_datetime]  # {area_id: {area/poi/land/link: [fact]}}
    temp_attr_used_list = temp_attr_used.split('-')

    # dict_final_temp
    subdict_temp_kge = {}
    # generate temp embeddings
    random_integer = random.randint(1, 1000)
    for _dim2 in range(x_goal.shape[1]):  # num_nodes
        # area_str = 'area_' + str(int(x_auxi[_dim1, _dim2, _dim3, 3]))
        area_str = 'area_' + str(int(x_auxi[0, _dim2, 3]))
        # temp embedding, 3*embed: 1-itself, 2-cont, 3-link
        temp_embed_1 = np.zeros(embed_dim, dtype=np.float)
        temp_embed_2 = np.zeros(embed_dim * 3, dtype=np.float)  # time/jam/weather
        embed2_time = np.zeros(embed_dim, dtype=np.float)
        embed2_jam = np.zeros(embed_dim, dtype=np.float)
        embed2_weather = np.zeros(embed_dim, dtype=np.float)
        temp_embed_3 = np.zeros(embed_dim * 3, dtype=np.float)
        embed3_hour = np.zeros(embed_dim, dtype=np.float)
        embed3_day = np.zeros(embed_dim, dtype=np.float)
        embed3_week = np.zeros(embed_dim, dtype=np.float)
        _subkg_fact = []
        if 'time' in temp_attr_used_list:
            _subkg_fact.extend(subdict_temp[area_str]['time'])
        if 'jam' in temp_attr_used_list:
            _subkg_fact.extend(subdict_temp[area_str]['jam'])
        if 'weather' in temp_attr_used_list:
            _subkg_fact.extend(subdict_temp[area_str]['tprt'])
            _subkg_fact.extend(subdict_temp[area_str]['rain'])
            _subkg_fact.extend(subdict_temp[area_str]['wind'])
        _subkg_fact_temp = []
        _subkg_fact_link = []
        for _fact in _subkg_fact:  # temp link
            if 'Hourly' in _fact[1] or 'Daily' in _fact[1] or 'Weekly' in _fact[1]:
                _subkg_fact_link.append(_fact)
            else:  # temp non_link
                _subkg_fact_temp.append(_fact)
        # embed 1
        if temp_model_used == 'ComplEx':
            temp_embed_1 = np.real(np.array(temp_ent_kge[temp_ent_label[area_str], :]))
        else:
            temp_embed_1 = np.array(temp_ent_kge[temp_ent_label[area_str], :])
        # embed 2
        for _fact in _subkg_fact_temp:
            _weight = float(_fact[3])
            if 'hourOfDay' in _fact[1] or 'dayOfWeek' in _fact[1] or 'isRestWork' in _fact[1]:
                embed2_time += cal_ent_via_rel(config, logger, temp_model_used, temp_ent_kge[temp_ent_label[_fact[2]], :],
                                              temp_rel_kge[temp_rel_label[_fact[1]], :], _weight)
            if 'hasjamCurrent' in _fact[1] or 'hasjamAve' in _fact[1]:
                embed2_jam += cal_ent_via_rel(config, logger, temp_model_used, temp_ent_kge[temp_ent_label[_fact[2]], :],
                                             temp_rel_kge[temp_rel_label[_fact[1]], :], _weight)
            if 'hastprtAve' in _fact[1] or 'hasrainAve' in _fact[1] or 'haswindAve' in _fact[1] or \
                    'hastprtCurrent' in _fact[1] or 'hasrainCurrent' in _fact[1] or 'haswindCurrent' in _fact[1]:
                embed2_weather += cal_ent_via_rel(config, logger, temp_model_used, temp_ent_kge[temp_ent_label[_fact[2]], :],
                                                 temp_rel_kge[temp_rel_label[_fact[1]], :], _weight)
        temp_embed_2[: embed_dim] = embed2_time
        temp_embed_2[embed_dim: 2 * embed_dim] = embed2_jam
        temp_embed_2[2 * embed_dim: 3 * embed_dim] = embed2_weather
        # embed 3
        for _fact in _subkg_fact_link:
            _weight = float(_fact[3])
            if 'Hourly' in _fact[1]:
                embed3_hour += cal_ent_via_rel(config, logger, temp_model_used, temp_ent_kge[temp_ent_label[_fact[2]], :],
                                             temp_rel_kge[temp_rel_label[_fact[1]], :], _weight)
            if 'Daily' in _fact[1]:
                embed3_day += cal_ent_via_rel(config, logger, temp_model_used, temp_ent_kge[temp_ent_label[_fact[2]], :],
                                             temp_rel_kge[temp_rel_label[_fact[1]], :], _weight)
            if 'Weekly' in _fact[1]:
                embed3_week += cal_ent_via_rel(config, logger, temp_model_used, temp_ent_kge[temp_ent_label[_fact[2]], :],
                                             temp_rel_kge[temp_rel_label[_fact[1]], :], _weight)
        temp_embed_3[: embed_dim] = embed3_hour
        temp_embed_3[embed_dim: 2 * embed_dim] = embed3_day
        temp_embed_3[2 * embed_dim: 3 * embed_dim] = embed3_week
        final_embed = np.concatenate((temp_embed_1, temp_embed_2, temp_embed_3), axis=0)
        subdict_temp_kge[area_str] = final_embed
        if _dim2 == random_integer and temp_datetime.time() == time(0, 5, 0):
            logger.info('{}, {} temp_embed1:{}'.format(temp_datetime, area_str, temp_embed_1))
            logger.info('{}, {} temp_embed2:{}'.format(temp_datetime, area_str, temp_embed_2))
            logger.info('{}, {} temp_embed3:{}'.format(temp_datetime, area_str, temp_embed_3))
    return subdict_temp_kge


def convert2datetime(x_auxi_slice):
    part1, part2, total_length = map(int, x_auxi_slice)
    long_num = part1 * 10 ** (total_length - len(str(part1))) + part2
    return datetime.fromtimestamp(long_num)


def process_temp_datetime_notcover(config, logger, temp_datetime, temp_datetime_list, np_goal, np_auxi, dictKG_temporal, dict_kge, kg_weight_temp):
    subdict_temp = generate_kgsub_temp_notcover(config, logger, dictKG_temporal, temp_datetime, temp_datetime_list, kg_weight_temp)
    dict_kge['sub_temp'][temp_datetime] = copy.deepcopy(subdict_temp)
    subdict_temp_kge = obtain_temp_kge_final(config, logger, np_goal, np_auxi, dict_kge, temp_datetime)
    dict_kge['sub_temp_emd'][temp_datetime] = copy.deepcopy(subdict_temp_kge)
    if temp_datetime.time() == time(0, 5, 0):
        logger.info('[During MP Complete]:{}'.format(temp_datetime))
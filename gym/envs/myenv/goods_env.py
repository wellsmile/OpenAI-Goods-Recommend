import gym
from gym import error, spaces, utils
from gym.utils import seeding

import random
from pprint import pprint
import numpy as np

import tensorflow as tf

def build_probability(probability_num):
    # 根据传入数值按概率返回T/F，通常概率为1-99之间‘
    probability_num = int(probability_num)
    if probability_num >= 100:
        return True
    if probability_num <= 0:
        return False
    a = random.randint(-(100-probability_num),probability_num)
    return True if a >= 0 else False
    
    
class Merchandise(object):
    def __init__(self, kind=None):
        if kind:
            self.kind = kind
        else:
            self.kind = random.choice(['book', 'car_supplies', 'keyboard', 'clothes', 'cosmetics', 'others'])
        self.money = random.randint(100,1000)
    
class User(object):
    def __init__(self):
        self.click_num = {'book': 0, 'car_supplies': 0, 'keyboard': 0, 'clothes': 0, 'cosmetics': 0, 'others': 0} # 每类商品点击次数dict
        self.gender = random.choice(['man', 'woman']) # 性别
        self.identity = random.choice(['student', 'worker', 'geek' ,'manager']) # 身份
        self.age = random.choice([18, 25, 35, 45, 60]) # 年龄
        self.average_pay = random.choice([100, 1000, 10000]) # 平均付费额
        self.last_click = random.choice(['book', 'car_supplies', 'keyboard', 'clothes', 'cosmetics', 'others']) # 上次点击商品类别
        
        self.total_money = 0
        self.total_num = 0
    
        self.build_id()
    
    def build_id(self):
        max_value = max(list(self.click_num.values()))
        max_key = list(self.click_num.keys())[list(self.click_num.values()).index(max_value)]
        self.id = '_'.join([max_key, self.gender, self.identity, str(self.age), str(self.average_pay), self.last_click])
        
    def build_choose_base(self):
        # 构建用户进行自主点击时的选择基础
        choose_base =  {'book': 5, 'car_supplies': 5, 'keyboard': 5, 'clothes': 5, 'cosmetics': 5, 'others': 5}
        
        # 模拟性别的影响
        if self.gender == 'man':
            choose_base['car_supplies'] += 1 if build_probability(80) else 0
            choose_base['cosmetics'] -= 1 if build_probability(80) else 0
        elif self.gender == 'woman':
            choose_base['clothes'] += 1 if build_probability(80) else 0
            choose_base['cosmetics'] += 1 if build_probability(80) else 0
            choose_base['car_supplies'] -= 1 if build_probability(80) else 0
        
        # 模拟身份的影响
        if self.identity == 'student':
            choose_base['book'] += 1 if build_probability(80) else 0
            choose_base['car_supplies'] -= 1 if build_probability(80) else 0
        elif self.identity == 'worker':
            choose_base['car_supplies'] += 1 if build_probability(80) else 0
            choose_base['book'] -= 1 if build_probability(80) else 0
        elif self.identity == 'geek':
            choose_base['book'] += 1 if build_probability(80) else 0
            choose_base['keyboard'] += 1 if build_probability(80) else 0
            choose_base['cosmetics'] -= 1 if build_probability(80) else 0
        elif self.identity == 'manager':
            choose_base['car_supplies'] += 1 if build_probability(80) else 0
            choose_base['clothes'] += 1 if build_probability(80) else 0
        
        #模拟年龄的影响
        if self.age <= 22:
            choose_base['book'] += 1 if build_probability(80) else 0
            choose_base['car_supplies'] -= 1 if build_probability(80) else 0
        elif self.age <= 40:
            choose_base['clothes'] += 1 if build_probability(80) else 0
            choose_base['car_supplies'] += 1 if build_probability(80) else 0
            choose_base['cosmetics'] += 1 if build_probability(80) else 0
        else:
             choose_base['book'] -= 1 if build_probability(80) else 0
             choose_base['cosmetics'] -= 1 if build_probability(80) else 0
             choose_base['keyboard'] -= 1 if build_probability(80) else 0
         
        # 历史点击总次数分布影响
        for key in choose_base.keys():
            choose_base[key] += int(self.click_num.get(key, 0)/2)
        
        # 上次点击商品分类影响
        choose_base[self.last_click] += 2 if build_probability(80) else 0
        
        self.choose_base = choose_base
    
    def choose(self):
        # 用户进行自主行动
        click = False
        pay = False
        choose_base_tmp = self.choose_base.copy()
        merchandise = Merchandise()
        while choose_base_tmp and not click:
            max_value = max(list(choose_base_tmp.values()))
            max_key = list(choose_base_tmp.keys())[list(choose_base_tmp.values()).index(max_value)]
            merchandise = Merchandise(max_key)
            choose_base_tmp.pop(max_key)
            if build_probability(75):
                self.click(merchandise)
                click = True
                if build_probability(25):
                    self.pay(merchandise)
                    pay = True
            self.build_id()
        return merchandise, click, pay
        
    def click(self, merchandise):
        # 用户点击商品
        self.last_click = merchandise.kind
        self.choose_base[merchandise.kind] += 1
        self.click_num[merchandise.kind] += 1
    
    def pay(self, merchandise):
        # 用户购买商品
        self.total_money += merchandise.money
        self.total_num += 1
        self.average_pays = float(self.total_money / self.total_num)
        self.choose_base[merchandise.kind] += 2
    
class GoodsEnv(gym.Env):
    def __init__(self):
        # 使用自定义的随机动作，要改为从多个商品中选择一个待推荐商品的动作
        self.action_space = spaces.Discrete(6)
#         self.observation_space = spaces.Discrete(99999999999999999)
        self.count = 0
        self.part_count = 0
        self.success = 0.0
        self.part_success = 0.0
        self._reset()
        self.action_dict = {}

#    seed方法不是必需的
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        
    def _step(self, action):
        if self.done:
            self._reset()
        # 智能体给用户推荐商品的行为
        # 推进一个步长的同时，需要用户进行点击和购买操作，然后智能体根据当前状态为用户推荐商品
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
#         choose_base = self.user.choose_base
#         merchandise_list = sorted(choose_base)
#         self.observation = int('0'.join([str(choose_base.get(choose_base_key)) for choose_base_key in merchandise_list]))
        self.observation = self.user.id
        merchandise_list = ['book', 'car_supplies', 'keyboard', 'clothes', 'cosmetics', 'others']
        merchandise_list.sort()
        
        # 模拟智能体
        action_dict_now = {}
        for action_choice in self.action_dict:
            if str(self.observation) in action_choice:
                action_dict_now[action_choice.split('=')[1]]  = self.action_dict[action_choice]
        for action_num in range(6):
            if str(action_num) not in action_dict_now.keys():
                action_dict_now[str(action_num)] = 0.0
        # 概率控制
        action_dict_control = action_dict_now.copy()
        for action_dict_now_key in action_dict_control:
            action_dict_control[action_dict_now_key] = action_dict_control[action_dict_now_key] + 16.6
        choosen = True
        while choosen:
            choosen_action = action_dict_control.copy()
            while choosen_action:
                max_value = max(list(choosen_action.values()))
                max_key = list(choosen_action.keys())[list(choosen_action.values()).index(max_value)]
                choosen_action.pop(max_key)
                if build_probability(max_value):
                    action = int(max_key)
                    choosen = False
        
        choice_recommend = merchandise_list[action]
        merchandise, click, pay = self.user.choose()
        
        reward = 0.0
        if choice_recommend == merchandise.kind:
            reward += 1.0 if click else 0.0
            reward += 1.0 if pay else 0.0
            self.success += 1
            self.part_success += 1
        else:
            reward -= 1.0
            
        # 训练一定次数，视为行为结束
        self.recommend_count += 1
        self.done = (self.recommend_count > 200)
        
        # 构建行为函数表示的dict
        action_dict_key = '='.join([str(self.observation), str(action)])
        action_dict_value = self.action_dict.get(action_dict_key, None)
        if action_dict_value:
            self.action_dict[action_dict_key] += reward
        else:
            self.action_dict[action_dict_key] = reward
        
        self.part_count += 1
        self.count += 1
        if self.count % 1000 == 0:
            self.part_count = 1
            self.part_success = 1 if choice_recommend == merchandise.kind else 0
        
        return self.observation, reward, self.done, {'now_recommend': choice_recommend,
                                                                      'now_hope': merchandise.kind,
                                                                      'now_control': action_dict_control,
                                                                      'count': self.count,
                                                                      'success': "%.2f%%" % ((self.success/self.count) * 100),
                                                                      'part_success:': "%.2f%%" % ((self.part_success/self.part_count) * 100)}
        
    def _reset(self):
        # 一个新的用户
        self.user = User()
        self.user.build_choose_base()
        self.done = False
        self.recommend_count = 0
        self.observation = 111111
        return self.observation
    
if __name__ == '__main__':
    user = User()
    user.build_choose_base()
    
    pprint(vars(user))
    

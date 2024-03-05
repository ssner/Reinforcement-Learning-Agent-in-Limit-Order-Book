# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 2020

@author: Xinyi
"""
xx =[8,7,6,5,4,3,2,1,0]
yy = [10**x for x in xx]
import numpy as np
class Agent:
    
    def __init__(self,enter_r,info_lag,price_adjust,informed,learning,ag_n,FV):
        self.enter_r = enter_r
        self.info_lag = info_lag
        self.price_adjust = price_adjust
        self.informed = informed
        self.learning = learning
        self.enter_countdown = np.random.randint(60)+1
        self.hold = 0
        self.last_action = []
        self.ag_n = ag_n
        self.buy_pro = 0.5
        self.MO_pro = 0.5
        self.profit = 0
        self.T = 1 #只是初始值 会更新
        self.oldT=0
        self.D = 0
        self.O = 0
        self.wherefisO = 0
        self.BS_values = []
        self.BS_times = []
        self.random_BS = 0
        self.OT_values = []
        self.OT_times = []
        self.random_OT = 0
        self.P = FV[0]
        if ((self.informed == 0) & (self.learning == 1)):
            self.pv1 = 0.12
            self.pv2 = 1
            self.pv3 = 2
            self.pv4 = 3
            self.pv5 = 4
            self.pv6 = 5
        else:
            self.pv1 = 0
            self.pv2 = 0
            self.pv3 = 0
            self.pv4 = 0
            self.pv5 = 0
            self.pv6 = 0
        
    def condition_building(self,spread_condition,best_bid,best_ask,last_bid,last_ask,last_trade_D,last_trade_P,depth_bid,depth_ask,depth_buy,depth_sell,Pav,FV,t):

        # 1 spread condition 2 bit: 
        # i.不为空： [3 0] [2 0] [1 0] ii.bid ask 都空: [0 1], bid空: [0,2],sell空:[0,3]
        self.spread_condition = spread_condition
        
        # 2 FV vs midprice 1 bit (有+1 -1 0 三种情况 暂时先这样计算 需要考虑一下agent交易完后的fundamental算第t期还是 第t+1期)
        # 对uninformed来说self.info_lag是300 对informed来说self.info_lag是0
        EFV = FV[max(t - self.info_lag-1,0)]
        if self.spread_condition[1] ==0:
           self.midprice = (best_bid + best_ask)/2
        elif self.spread_condition[1] ==1:
            self.midprice = last_trade_P
        elif self.spread_condition[1] == 2:
            self.midprice = best_ask - self.price_adjust
        elif self.spread_condition[1] == 3:
            self.midprice = best_bid + self.price_adjust
        self.midP_condition = int(np.sign(self.midprice - EFV)) + 1
        
        # 3.Rosu's signal 1 bit (1 = < 0.5, 2 = <1.5, 3 = <2.5, 4 = < 3.5, 5 = > =3.5)
        if self.informed == 1:
            EFV = FV[t-1]
        else:
            EFV = Pav
        #??
        if self.spread_condition[0] == 0:
            if EFV > best_ask or EFV < best_bid:
                self.rosu_signal = 5
            else:
                self.rosu_signal = 1
        else:
            signal = abs(EFV-self.midprice)/(best_ask - best_bid)
            if signal < 0.5:
                self.rosu_signal = 1
            elif signal < 1.5:
                self.rosu_signal  = 2
            elif signal <2.5:
                self.rosu_signal = 3
            elif signal < 3.5:
                self.rosu_signal = 4
            elif signal >= 3.5:
                self.rosu_signal = 5
        
        #4. current bid versus last bid 1 bit (+1 -1 0)
        #5 current ask versus last ask 1 bit (+1 -1 0)
        if (best_bid != float('inf'))&(best_bid != float('-inf'))&(last_bid != float('inf'))&(last_bid != float('-inf')):
            self.bid_condition = int(np.sign(best_bid - last_bid)) + 1
        else:
            self.bid_condition = 0 + 1
        if (best_ask != float('inf'))&(best_ask != float('-inf'))&(last_ask != float('inf'))&(last_ask != float('-inf')):
            self.ask_condition = int(np.sign(best_ask - last_ask)) + 1
        else:
            self.ask_condition = 0 + 1
        
        
        #6. ask depth versus bid depth 1 bit (+1 -1 0)
        #7. cumulative sell depth versus cumulative buy depth 1 bit (+1 -1 0)
        self.depth_condition = np.sign(depth_ask - depth_bid) + 1
        self.cumulative_depth_con = np.sign(depth_sell - depth_buy) + 1
        
        #8 last trade direction 1 bit(+1 buy -1 sell) 0 2
        self.last_trade_D  = last_trade_D+1
        
        self.Con = self.spread_condition + [ self.midP_condition, self.rosu_signal, self.bid_condition, self.ask_condition, self.depth_condition,self.cumulative_depth_con,self.last_trade_D]

        self.Con_scalar = int(np.sum(list(map(lambda x :x[0]*x[1] ,zip(self.Con,yy)))))
        return self.Con,self.Con_scalar#其实好像可以不用return?
 
    def action_selection(self, FV, t,last_trade_P,Pav,UNQ_BS_values,UNQ_OT_values,\
                         INQ_BS_values,INQ_OT_values,epsilon1,epsilon2):
        #ConTuple = tuple(self.Con)
        
        # 1.noise trader action(uninformed not learn)
        if (self.learning == 0) & (self.informed == 0):
            # noise trader 选择买卖或no trade
            #self.D = np.random.randint(-1,2)
            self.D = int(np.sign(FV[max(t - self.info_lag-1,0)] - self.midprice))
            #self.D = np.sign(Pav - self.midprice)
            # 1 market order at 2 extremely agressive limit order at - 1 3 agressive limit order bt + 1 4. Normal limit order bt 
            # 5. Unagressive limit order bt - 1  
            if self.D != 0:
                va = [1, 2, 3, 4, 5]
                if self.Con[0] == 1:
                    del va[1:3]
                if self.Con[0] == 2:
                    del va[2]
                self.O = va[np.random.randint(len(va))]
            else:
                self.O = 0
            
        # 2. informed not learn
        if (self.learning == 0) & (self.informed ==  1):
            # informed选择买卖或no trade 根据基本面价格和上次交易价格之差
            self.D = int(np.sign(FV[t-1] - self.midprice))
            # 1 market order at 2 extremely agressive limit order at - 1 3 agressive limit order bt + 1 4. Normal limit order bt 
            # 5. Unagressive limit order bt - 1  
            if self.D != 0:
                va = [1, 2, 3, 4, 5]
                if self.Con[0] == 1:
                    del va[1:3]
                if self.Con[0] == 2:
                    del va[2]
                self.O = va[np.random.randint(len(va))]
            else:
                self.O = 0
                
        # 3. uninformed learning
        if (self.learning == 1) & (self.informed == 0):

            # Buy and sell determination
            self.BS_values = UNQ_BS_values[UNQ_BS_values[:,0] == self.Con_scalar,1:4].ravel()
            #self.BS_times = UNREC_BS_times[UNREC_BS_times[:,0] == self.Con_scalar,1:4].ravel()
            self.random_BS = np.random.random()
            if (self.random_BS <= epsilon1) or (all(self.BS_values==0)):
                self.D = np.random.randint(-1,2)
            elif (self.random_BS >epsilon1):
                self.D = np.argmax(self.BS_values) - 1
            
            # Order type determination
            if self.D != 0:
                self.OT_values = UNQ_OT_values[UNQ_OT_values[:,0] == self.Con_scalar,int(((self.D+1)/2)*5+1):int(((self.D+1)/2)*5+6)].ravel()
               # OT_times = UNREC_OT_times[UNREC_OT_times[:,0] == self.Con_scalar,int(((self.D+1)/2)*5+1):int(((self.D+1)/2)*5+6)].ravel()
                va = [1,2,3,4,5]
                if self.Con[0] == 1:
                    del va[1:3]
                    self.OT_values = np.delete(self.OT_values,[1,2])
                    #OT_times = np.delete(OT_times,[0,1])
                if self.Con[0] == 2:
                    del va[2]
                    self.OT_values = np.delete(self.OT_values,2)
                    #OT_times = np.delete(OT_times,0)
                self.random_OT = np.random.random()
                if (self.random_OT <= epsilon2) or (all(self.OT_values==0)):
                    self.O = va[np.random.randint(len(va))]
                    self.wherefisO = 1
                if (self.random_OT > epsilon2):
                    self.O = va[np.argmax(self.OT_values)]
                    self.wherefisO = 2
            else:  
                self.O = 0
                self.wherefisO = 3
        
        # 4. informed learning
        if (self.learning == 1) & (self.informed == 1):

            # Buy and sell determination
            
            self.BS_values = INQ_BS_values[INQ_BS_values[:,0] == self.Con_scalar,1:4].ravel()
            #self.BS_times = INREC_BS_times[INREC_BS_times[:,0] == self.Con_scalar,1:4].ravel()
            self.random_BS = np.random.random()
            if (self.random_BS <= epsilon1) or (all(self.BS_values==0)):
                self.D = np.random.randint(-1,2)
            elif (self.random_BS > epsilon1):
                self.D = np.argmax(self.BS_values) - 1
            
            # Order type determination
            if self.D != 0:
                OT_values = INQ_OT_values[INQ_OT_values[:,0] == self.Con_scalar,int(((self.D+1)/2)*5+1):int(((self.D+1)/2)*5+6)].ravel()
               # OT_times = INREC_OT_times[INREC_OT_times[:,0] == self.Con_scalar,int(((self.D+1)/2)*5+1):int(((self.D+1)/2)*5+6)].ravel()
                va = [1,2,3,4,5]
                if self.Con[0] == 1:
                    del va[1:3]
                    OT_values = np.delete(OT_values,[1,2])
                    #OT_times = np.delete(OT_times,[0,1])
                if self.Con[0] == 2:
                    del va[2]
                    OT_values = np.delete(OT_values,2)
                    #OT_times = np.delete(OT_times,0)
                self.random_OT = np.random.random()
                if (self.random_OT <= epsilon2) or (all(OT_values==0)):
                    self.O = va[np.random.randint(len(va))]
                elif (self.random_OT > epsilon2):
                    self.O = va[np.argmax(OT_values)]
            else:  
                self.O = 0
                  
        return self.D,self.O #可删
# t代表期数,x代表第几个进入市场的agent!!!!!!!    
    def order_building(self,t,x,best_bid,best_ask,last_trade_P):
        self.T = t
        self.OrderID = t * (10**3) + x+1
        
        if self.D != 0:
            if self.Con[1] == 1:
                MPB = last_trade_P - self.price_adjust
                MPS = last_trade_P + self.price_adjust
            elif self.Con[1] == 2:
                MPB = best_ask - self.price_adjust
                MPS = best_ask
            elif self.Con[1] == 3:
                MPB = best_bid
                MPS = best_bid + self.price_adjust
            else:
                MPB  = best_bid
                MPS = best_ask
            if self.D == 1:
                if self.O == 1:
                    self.P = MPS
                elif self.O ==2:
                    self.P = MPS - 1
                elif self.O ==3:
                    self.P = MPB + 1
                elif self.O == 4:
                    self.P = MPB
                elif self.O == 5:
                    self.P = MPB - 1
            if self.D == -1:
                if self.O == 1:
                    self.P = MPB
                elif self.O == 2:
                    self.P = MPB + 1
                elif self.O == 3:
                    self.P = MPS - 1
                elif self.O == 4:
                    self.P = MPS
                elif self.O ==  5:
                    self.P = MPS + 1
                
            self.Order = [self.D, self.P]
        else:
            self.Order = []
        return self.Order, self.OrderID

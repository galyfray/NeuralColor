# Hello ! 
# this file is my first attempt at creating a neural system for machine learning
# this exemple is rather simple as the system is meant to predict if an RGB code correspond to pink
# this system is based on the simplest neuron system I found formal neurons
# this program is only meant to work and I will create a much proper version later on another repo.




from enum import Enum
import random
from abc import ABC, abstractmethod

# RGB code range for a color here pink

R = [255,145]
G = [234,40]
B = [225,59]

class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3

# basic neuron

class neuralProvider(ABC):
    
    def __init__(self):
        self.recivers = []
        
    def register(self,reciver):
        self.recivers.append(reciver)
        
    @abstractmethod
    def send(self,data:int,sender):
        pass

# a classic formalNeuron

class formalNeurons(neuralProvider):
    
    def __init__(self,providers,weightList,ceil,register=True):
        neuralProvider.__init__(self)
        
        assert len(weightList) == len(providers)
        
        self.providers = {providers[i]:{"weight":weightList[i],"data":None} for i in range(len(weightList))}
        
        self.ceil = ceil
        
        if register :
            for provider in providers:
                provider.register(self)
        
    def send(self,data:int,sender:neuralProvider):
        if (self.providers[sender]["data"] == None):
            self.providers[sender]["data"] = data
        else :
            raise RuntimeError(f"Sender {sender} has sent data twices")
    
    def compute(self):
        sum = 0
        for val in self.providers.values():
            if (val["data"] == None):
                raise RuntimeError("computation started while data are still missing")
            sum += val["data"] * val["weight"]
            val["data"] = None
        
        for reciver in self.recivers:
            reciver.send(1 if sum >= self.ceil else 0,self)

# Input neuron create data check if it is the right color

class RGBProvider(neuralProvider):
    
    def __init__(self,RRange,GRange,BRange):
        self.RRange=RRange
        self.GRange=GRange
        self.BRange=BRange
        self.recivers = {c:[] for c in Color}
        self.isInRange = None
    
    def register(self,reciver,color:Color):
        self.recivers[color].append(reciver)
    
    def sendToSystem(self):
        r = random.randint(0,255)
        g = random.randint(0,255)
        b = random.randint(0,255)
        
        self.isInRange = r<=self.RRange[0] and r>=self.RRange[1]
        self.isInRange = g<=self.GRange[0] and g>=self.GRange[1] and self.isInRange
        self.isInRange = b<=self.BRange[0] and b>=self.BRange[1] and self.isInRange
        
        for red in self.recivers[Color.RED]:
            red.send(r,self)
        
        for green in self.recivers[Color.GREEN]:
            green.send(g,self)
        
        for blue in self.recivers[Color.BLUE]:
            blue.send(b,self)
    
    def send(self, data, sender):
        pass

# Allow easier creation of a neural system 

class neuralSystem(object):
    
    def __init__(self,neuralDict=None):
        if neuralDict == None :
            neuralDict = {"Inputs":{"n":0},"Processors":{}}
        self.neuralDict = neuralDict
        self.system = {"Inputs":{},"Processors":{}}
        self.lastStrat = -1
    
    def addInput(self,weight,ceil):
        I = self.neuralDict["Inputs"]
        I[I["n"]] = {"weightList" : [weight],"ceil":ceil}
        I["n"]+=1
    
    def addProcessor(self,strat,weightList,ceil):
        P = self.neuralDict["Processors"]
        
        if self.lastStrat < strat:
            self.lastStrat = strat
        
        if not strat in P.keys():
            P[strat] = {"n":0}
        P = P[strat]
        P[P["n"]] = {"weightList" : weightList,"ceil":ceil}
        P["n"]+=1
    
    def build(self,startingPoint:neuralProvider):
        del self.neuralDict["Inputs"]["n"]
        
        for n,data in self.neuralDict["Inputs"].items() :
            self.system["Inputs"][n] = formalNeurons([startingPoint],data["weightList"],data["ceil"],False)
        
        for s in range( self.lastStrat +1):
            
            del self.neuralDict["Processors"][s]["n"]
            self.system["Processors"][s] = {}
            
            prev = []
            
            if s == 0:
                prev = [v for v in self.system["Inputs"].values()]
            else :
                prev = [v for v in self.system["Processors"][s-1].values()]
                
            for n,data in self.neuralDict["Processors"][s].items():
                self.system["Processors"][s][n] = formalNeurons(prev,data["weightList"],data["ceil"])
    
    def run(self):
        for elem in self.system["Inputs"].values():
            elem.compute()

        for strat in self.system["Processors"].values():
            for elem in strat.values():
                elem.compute()
    
    def getOutput(self):
        return self.system["Processors"][self.lastStrat][0]

# statistique making neuron

class StatMaker(neuralProvider):
    
    def __init__(self,provider):
        self.stat = {}
        self.provider = provider
        self.best = None
        self.bestScore = 0
    
    def send(self,data:int,sender):
        v = 1 if self.provider.isInRange else 0 
        if sender in self.stat.keys():
            self.stat[sender].append(v == data)
        else:
            self.stat[sender] = [v == data]
    
    def processStat(self,systems):
        print("=================Stat Time !==================")
        s = None
        n = 0
        for k,v in enumerate(self.stat.keys()):
            a = 0
            
            for t in self.stat[v] :
                a += 1 if t else 0
            a/= len(self.stat[v])
            
            if a > self.bestScore :
                self.bestScore = a
                s = v
                n = k
            print(f"System {k} : efficiency : {100*a} % ")
        
        for system in systems :
            if system.getOutput() == s:
               self.best = system
        
        
        print("###################################")
        print(f"Best System : {n} | {100*self.bestScore} % Topography :\n {self.best.neuralDict}")
        print("=================== End ======================")      
        
        
RGB = RGBProvider(R,G,B)

stat = StatMaker(RGB)

poolSize = 100
pool = []

#inital pool of random neuron system

for _ in range(poolSize) :

    system = neuralSystem()

    system.addInput(random.random(),random.randint(0,255)) #R
    system.addInput(random.random(),random.randint(0,255)) #G
    system.addInput(random.random(),random.randint(0,255)) #B
    
    system.addProcessor(0,[random.random()*3,random.random()*3,random.random()*3],random.randint(0,10))
    system.addProcessor(0,[random.random()*3,random.random()*3,random.random()*3],random.randint(0,10))
    system.addProcessor(0,[random.random()*3,random.random()*3,random.random()*3],random.randint(0,10))
    system.addProcessor(0,[random.random()*3,random.random()*3,random.random()*3],random.randint(0,10))
    system.addProcessor(0,[random.random()*3,random.random()*3,random.random()*3],random.randint(0,10))
    
    system.addProcessor(1,[random.random()*3,random.random()*3,random.random()*3,random.random()*3,random.random()*3],random.randint(0,10))
    system.addProcessor(1,[random.random()*3,random.random()*3,random.random()*3,random.random()*3,random.random()*3],random.randint(0,10))
    system.addProcessor(1,[random.random()*3,random.random()*3,random.random()*3,random.random()*3,random.random()*3],random.randint(0,10))
    
    system.addProcessor(2,[random.random()*3,random.random()*3,random.random()*3],random.randint(0,10))
    
    system.build(RGB)
    
    pool.append(system)
    
    RGB.register(system.system["Inputs"][0],Color.RED)
    RGB.register(system.system["Inputs"][1],Color.GREEN)
    RGB.register(system.system["Inputs"][2],Color.BLUE)
    
    system.getOutput().register(stat)

for i in range(1000):
    RGB.sendToSystem()
    for sys in pool:
        sys.run()

stat.processStat(pool)

scores = [stat.bestScore]

bestScore = stat.bestScore

best = stat.best.neuralDict

#genetic algorythm to improove system performance

for _ in range(100):
    
    RGB = RGBProvider(R,G,B)

    stat = StatMaker(RGB)
    
    pool = []
    
    for _ in range(poolSize) :

        system = neuralSystem()
        
        BI = best["Inputs"]
        for i in range(3):
            system.addInput(BI[i]["weightList"][0] + (random.random() - 0.5)
                            ,random.randint(-10,10) + BI[i]["ceil"]) #R
        
        for i in range(5) :
            w = [v + (random.random() - 0.5)*2 for v in best["Processors"][0][i]["weightList"]]
            system.addProcessor(0,w,best["Processors"][0][i]["ceil"] + random.randint(-1,1))
            
        for i in range(3) :
            w = [v + (random.random() - 0.5)*2 for v in best["Processors"][1][i]["weightList"]]
            system.addProcessor(1,w,best["Processors"][1][i]["ceil"] + random.randint(-1,1))
        
        w = [v + (random.random() - 0.5)*2 for v in best["Processors"][2][0]["weightList"]]
        system.addProcessor(2,w,best["Processors"][2][0]["ceil"] + random.randint(-1,1))
        
        system.build(RGB)
        
        pool.append(system)
        
        RGB.register(system.system["Inputs"][0],Color.RED)
        RGB.register(system.system["Inputs"][1],Color.GREEN)
        RGB.register(system.system["Inputs"][2],Color.BLUE)
        
        system.getOutput().register(stat)
    
    for i in range(1000):
        RGB.sendToSystem()
        for sys in pool:
            sys.run()
    
    stat.processStat(pool)
    
    if bestScore < stat.bestScore :
        best = stat.best.neuralDict
        bestScore = stat.bestScore
    
    scores.append(stat.bestScore)
    
    
    
    
    
    
print(scores)
print(bestScore)
print(best)
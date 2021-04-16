# RGB code range for a color here 

from enum import Enum
import random
from abc import ABC, abstractmethod

R = [255,145]
G = [234,40]
B = [225,59]

class neuralProvider(ABC):
    
    def __init__(self):
        self.recivers = []
        
    def register(self,reciver):
        self.recivers.append(reciver)
        
    @abstractmethod
    def send(self,data:int,sender):
        pass

class formalNeurons(neuralProvider):
    
    def __init__(self,providers,weightList,ceil,register=True):
        neuralProvider.__init__(self)
        
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

class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3

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
            
class neuralSystem(object):
    
    def __init__(self,neuralDict={"Inputs":{"n":0},"Processors":{}}):
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
            
RGB = RGBProvider(R,G,B)

system = neuralSystem()

system.addInput(1,145) #R
system.addInput(1,40) #G
system.addInput(1,59) #B

system.addProcessor(0,[1,1,1],3)

system.build(RGB)

RGB.register(system.system["Inputs"][0],Color.RED)
RGB.register(system.system["Inputs"][1],Color.GREEN)
RGB.register(system.system["Inputs"][2],Color.BLUE)

out = system.system["Processors"][0][0]

class StatMaker(neuralProvider):
    
    def __init__(self,provider):
        self.stat = {}
        self.provider = provider
    
    def send(self,data:int,sender):
        v = 1 if self.provider.isInRange else 0 
        if sender in self.stat.keys():
            self.stat[sender].append(v == data)
        else:
            self.stat[sender] = [v == data]
    
    def processStat(self,systems):
        print("=================Stat Time !==================")
        m = -1
        s = None
        n = 0
        for k,v in enumerate(self.stat.keys()):
            a = 0
            for t in self.stat[v] :
                a += 1 if t else 0
            a/= len(self.stat[v])
            if a > m :
                m = a
                s = v
                n = k
            print(f"System {k} : efficassiter : {100*a} % ")
        
        sys = None
        
        for system in systems :
            if system.getOutput() == s:
               sys = system
        
        
        print("###################################")
        print(f"Best System : {n} | {100*a} % Topography :\n {sys.neuralDict}")
        print("=================== End ======================")

stat = StatMaker(RGB)
out.register(stat)

for i in range(10):
    RGB.sendToSystem()
    system.run()

stat.processStat([system])


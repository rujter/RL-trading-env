#dejmo rect da je cena 0 do 100 (0-10001)
from collections import deque
import time
import random
import torch
import torch.nn as nn #neural network building blocks
import torch.optim as optim #oprimisers
import torch.nn.functional as F #functions like softmaxx

class TradingEnv:
    def __init__(self, starting_cash=1000.0, starting_price=100.0, trade_qty=10):
        self.starting_cash = starting_cash
        self.starting_price = starting_price
        self.trade_qty = trade_qty

    def reset(self):
        self.cash = self.starting_cash
        self.price = self.starting_price
        self.last_price = self.starting_price
        self.qty = 0
        return self._get_state()

    def _get_state(self):
        price_change = self.price - self.last_price
        return [self.price, price_change, self.cash, self.qty]

    def _portfolio_value(self):
        return self.cash + self.qty * self.price

    def step(self, action):
        value_before = self._portfolio_value()
        traded = False

        if action == 0:
            cost = self.trade_qty * self.price
            if self.cash >= cost:
                self.cash -= cost
                self.qty += self.trade_qty
                traded = True
        elif action == 1:
            if self.qty >= self.trade_qty:
                self.cash += self.trade_qty * self.price
                self.qty -= self.trade_qty
                traded = True
        # action == 2 (hold) intentionally does nothing -- traded stays False

        self.last_price = self.price
        direction = random.uniform(0, 1)
        magnitude = random.uniform(0, 0.1)
        if direction < 0.45:
            self.price *= (1 - magnitude)
        else:
            self.price *= (1 + magnitude)

        value_after = self._portfolio_value()
        reward = value_after - value_before

        if action != 2 and not traded:
            reward -= 1.0   # $$$$$$$$$$$$ ADDED: penalize attempting an invalid trade (e.g. selling with no shares)

        done = value_after < 0.10 * self.starting_cash
        return self._get_state(), reward, done
    
class Actor(nn.Module):
    def __init__(self, state_size, num_actions): #state size pove koliko parametrov ima state torej tipo zame ker bom imel
        #curr price, price change, cash, quantity

        #num actions je enako

        super().__init__() #to rabis ker pac poklice constructor in to rabis
        self.fc1 = nn.Linear(state_size, 64) #nn.Linear(in_features, out_features), takes your raw state (e.g. 4 or 5 numbers) → outputs 64 number
        self.fc2 = nn.Linear(64, 64) #pac to racuna, ce imas 3 state racuna z nekimi weights output = (price × w1) + (cash × w2) + (qty × w3) + bias
        self.fc3 = nn.Linear(64, num_actions) #64 different neurons, each one independently doing that same weighted-sum
        #vec layerjev vec complexitija

    def forward(self, state): #this happens when you acc call network actor
        x = F.relu(self.fc1(state)) #to moras imet da dejansko passas skozi te transformacije 
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)             
        return F.softmax(logits, dim=-1)   #converts those raw numbers into a proper probability distribution by softmax to sum to 1


class Critic(nn.Module):
    def __init__(self, state_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)         # single output: V(s)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)                  #samo en number ker estimatea kaj se je zgodilo


def select_action(actor, state):
    state_tensor = torch.tensor(state, dtype=torch.float32)

    probs = actor(state_tensor)                       #dobis nazaj probabilitije 
    dist = torch.distributions.Categorical(probs)       #basically izbere weighted random
    action = dist.sample()                               # actually picks one, respecting the probabilities

    log_prob = dist.log_prob(action)                     # log(π(a|s)) -- you'll need this for the actor's loss later
    return action.item(), log_prob



def train_step(actor, critic, actor_optimizer, critic_optimizer, state, action, log_prob, reward, next_state, done, gamma=0.99):
    state_t = torch.tensor(state, dtype=torch.float32) #nov float tensor
    next_state_t = torch.tensor(next_state, dtype=torch.float32) #nov float tensor

    value = critic(state_t) #dobi number value iz critica
    next_value = critic(next_state_t) if not done else torch.tensor([0.0]) #next value

    # this line IS your TD formula from your notes: δ = r + γ·V(s') - V(s)
    td_target = reward + gamma * next_value.detach()
    advantage = td_target - value   #was it better or wordse

    # critic loss: make V(s) closer to the TD target -- plain regression, "how wrong is the network right now"
    #given a state, predict how much total future reward you can expect from here
    critic_loss = advantage.pow(2).mean()

    # actor loss: policy gradient using the advantage as the signal
    actor_loss = -log_prob * advantage.detach()

    actor_optimizer.zero_grad() #AVG TRAING STEPS ZA USE KAR BOS RABU
    actor_loss.backward()
    actor_optimizer.step()

    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

env = TradingEnv()
state_size = 4    # price, price_change, cash, qty, (whatever else you defined)
num_actions = 3      # buy, sell, hold

actor = Actor(state_size, num_actions)
critic = Critic(state_size)
actor_optimizer = optim.Adam(actor.parameters(), lr=0.001)
critic_optimizer = optim.Adam(critic.parameters(), lr=0.001)

for episode in range(20):
    state = env.reset()
    done = False
    steps = 0
    episode_reward = 0

    while not done and steps < 500:

        action, log_prob = select_action(actor, state) #glede na actor izbere input
        if steps < 5:
            print("action chosen:", action)          # $$$$$$$$$$$$ ADD THIS LINE

        next_state, reward, done = env.step(action) #inpuit poslje v env in dobi nazaj next state,reward etc

        train_step(actor, critic, actor_optimizer, critic_optimizer, #na teh podatkih tradea 
                   state, action, log_prob, reward, next_state, done)

        state = next_state #spremeni state etc
        episode_reward += reward #SAMO ZA MOJ VISIBILITY NI POMEMBEN IN THE GRAND SCHEME OF THINGS
        steps += 1

    print(f"Episode {episode}: reward={episode_reward:.2f}, steps={steps}, done={done}")
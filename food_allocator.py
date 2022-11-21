import random
import sys
import re
from copy import deepcopy


class User:
    def __init__(self, food_dict, non_veg, non_diabetic_friendly):
        trigger_words = ['baby', 'climate', 'wastage']
        self.vegetarian = (random.uniform(0,1) < 0.3) # assuming ~30% people are vegetarian
        self.diabetic = (random.uniform(0,1) < 0.1) # assuming ~10% people are diabetic
        self.preferences = dict()
        for food_category in food_dict:
            pref_array = []
            for food_option in food_dict[food_category]:
                if self.vegetarian and food_dict[food_category][food_option] in non_veg:
                    continue
                elif self.diabetic and food_dict[food_category][food_option] in non_diabetic_friendly:
                    continue
                else:
                    pref_array.append(food_option)
            random.shuffle(pref_array)
            self.preferences[food_category] = pref_array.copy()
        
        # print('User preferences: ',self.preferences)
        self.reward = 0
        self.adjustments = 0
        self.replacements = 0
        self.trigger_words = []
        for i in range(1): # only one trigger word for now
            self.trigger_words.append(random.choices(trigger_words)[0])
        self.threshold = random.uniform(0.45,0.55)
        self.original_threshold = self.threshold

    def request(self, food_category):
        choice = random.choices(self.preferences[food_category])[0]
        return choice

    def get_response(self, food_category, offer):
        self.replacements += 1
        for i in range(len(self.preferences[food_category])):
            if self.preferences[food_category][i] == offer:
                pos = i
                break
        # print('pos: ',pos)
        resp = (len(self.preferences[food_category])-pos) * 1/len(self.preferences[food_category]) * random.uniform(0.85,1.15)
        print('User Response:',resp)
        if resp >= 0.5:
            self.reward += pos + 1 # adjustment is a factor of how much they prefer it (inversely proportional)
            self.adjustments += 1
        user_resp = 0
        # divide user response into sections- can think of this as how happy they are with the substitution
        if resp > 0.9: user_resp = 1
        elif resp > 0.7 : user_resp = 0.7
        elif resp >= 0.5: user_resp = 0.5
        elif resp > 0.3: user_resp = 0.3
        elif resp > 0.1: user_resp = 0.1
        else: user_resp = 0
        return user_resp

    def get_nudged(self, agent_feedback):
        # user gets nudged if their trigger word
        agent_feedback = set(re.split('[ !,.]', agent_feedback.lower()))
        for word in self.trigger_words:
            print('trigger:',word)
            if word in agent_feedback:
                print('User nudged by 0.1%')
                self.threshold *= 0.999
        
        



class FoodAgent:
    def __init__(self, user, food_dict, non_veg, non_diabetic_friendly):
        self.epsilon = sys.float_info.epsilon
        self.foods = food_dict
        self.user = user # user associated with this agent
        self.exp = 0.3
        self.dist = 0.2
        self.decay = 0.8
        self.rwt = 0.8
        self.w0 = 0.5
        self.options = dict()
        self.probs = dict()
        self.veg_mask = dict()
        self.diabetic_mask = dict()
        self.w = dict()
        for food_category in food_dict:
            self.options[food_category] = [i for i in range(len(self.foods[food_category]))]
            self.probs[food_category] = [1/len(food_category) for i in range(len(self.foods[food_category]))]
            self.w[food_category] = [self.w0 for i in range(len(food_category))]
            self.veg_mask[food_category] = []
            self.diabetic_mask[food_category] = []
            for food_item in food_dict[food_category]:
                if food_dict[food_category][food_item] in non_veg:
                    self.veg_mask[food_category].append(0)
                else:
                    self.veg_mask[food_category].append(1)
                if food_dict[food_category][food_item] in non_diabetic_friendly:
                    self.diabetic_mask[food_category].append(0)
                else:
                    self.diabetic_mask[food_category].append(1)
        if user.vegetarian:
            for food_category in food_dict:
                self.probs[food_category] = [(a*b)for a,b in zip(self.probs[food_category], self.veg_mask[food_category])]
                self.w[food_category] = [(a*b) for a,b in zip(self.w[food_category], self.veg_mask[food_category])]
        if user.diabetic:
            for food_category in food_dict:
                self.probs[food_category] = [(a*b)for a,b in zip(self.probs[food_category], self.diabetic_mask[food_category])]
                self.w[food_category] = [(a*b) for a,b in zip(self.w[food_category], self.diabetic_mask[food_category])]

        self.cum_rew = 0

    def suggest(self, food_category, stock):
        wts = [(a*b)+self.epsilon for a,b in zip(self.probs[food_category], stock)] # combination of user preferences and stock used to suggest to user
        if self.user.vegetarian:
            wts = [(a*b) for a,b in zip(wts, self.veg_mask[food_category])]
            # wts *= self.veg_mask[food_category]
        if self.user.diabetic:
            wts = [(a*b) for a,b in zip(wts, self.diabetic_mask[food_category])]
            # wts *= self.diabetic_mask[food_category]
        # print(wts)
        wts_sum = sum(wts)
        wts = [wt/wts_sum for wt in wts] # normalizing weights to use in making a suggestion
        suggested_option = random.choices(self.options[food_category], weights = wts)[0]
        return suggested_option

    def learn(self, food_category, suggested_option, user_response):
        # learn from user's response to the suggestion
        reward = user_response
        self.w[food_category][suggested_option] = self.decay * self.w[food_category][suggested_option] + self.rwt * reward
        normalized_w = self.normalized_weight(food_category, suggested_option)
        self.probs[food_category][suggested_option] = normalized_w*(1-self.exp) + self.dist * self.exp # update probability of chosen food item
        self.normalize_probs(food_category)
        self.cum_rew += reward

    def get_learnt_preferences(self):
        calculated_probs = dict()
        for food_category in self.foods:
            calculated_probs[food_category] = dict()
            for i,prob in enumerate(self.probs[food_category]):
                if prob == 0:
                    continue
                if prob in calculated_probs[food_category]:
                    calculated_probs[food_category][prob].add(i)
                else:
                    calculated_probs[food_category][prob] = set()
                    calculated_probs[food_category][prob].add(i) # i is index of food
        sorted_probs = dict()
        sorted_prefs = dict()
        for food_category in self.foods:
            sorted_probs[food_category] = list(set(self.probs[food_category]))
            sorted_probs[food_category] = sorted(sorted_probs[food_category], reverse = True)
            sorted_prefs[food_category] = []
        for food_category in self.foods:
            for prob in sorted_probs[food_category]:
                if prob == 0: continue
                if len(calculated_probs[food_category][prob]) == 1:
                    for food_index in calculated_probs[food_category][prob]:
                        sorted_prefs[food_category].append(self.foods[food_category][food_index])
                else: # many items with this same prob
                    temp = []
                    for food_index in calculated_probs[food_category][prob]:
                        temp.append(self.foods[food_category][food_index])
                    sorted_prefs[food_category].append(temp)
        return sorted_prefs

    def normalize_probs(self, food_category):
        min_prob = min(self.probs[food_category])
        if min_prob < 0: # removing negative probabilities
            self.probs[food_category] += abs(min_prob)
        p_sum = sum(self.probs[food_category])
        self.probs[food_category]  = [prob/p_sum for prob in self.probs[food_category]] # normalizing to add to 1

    def normalized_weight(self, food_category, choice):
        min_w = min(self.w[food_category])
        max_w = max(self.w[food_category])
        if max_w == min_w:
            # if max_w == 0:
            #     self.w = [self.w0 for i in range(self.n)]
            #     return self.w0
            return self.w[food_category][choice]/max_w
        return (self.w[food_category][choice]-min_w)/(max_w - min_w)


class FoodAllocator:
    def __init__(self,num_users):
        self.non_veg = {'chili beans with pork', 'black beans chili'}
        self.non_diabetic_friendly = {'grapes'}
        self.num_users = num_users # number of users
        self.foods = {'fruits': {0: 'apple', 1: 'orange', 2:'banana', 3:'pear', 4:'grapes', 5:'kiwi'},
                    'canned_items': {0: 'garbanzo beans', 1:'black eyed peas', 2: 'black beans', 3:'chili beans', 4:'chili beans with pork', 5:'black beans chili'}}
        self.n = len(self.foods)
        self.users = []
        self.user_agent_map = dict()
        self.good_feedback = ['This item was requested by a family for their baby, you helped them very much', 'Climate change is controlled by people like you, who are willing to adjust, thank you!', 
        'You helped prevent food wastage, thank you!']
        self.bad_feedback = ['A family requested this for their baby, please consider again next time', 'This cultivation is worse for climate change, please consider again next time', 
        'Some food wastage has occurred, please consider again next time']
        for i in range(self.num_users):
            new_user = User(self.foods, self.non_veg, self.non_diabetic_friendly) # new user with random preferences over the n items
            new_agent = FoodAgent(new_user, self.foods, self.non_veg, self.non_diabetic_friendly) # agent for this user created
            self.users.append(new_user)
            self.user_agent_map[new_user] = new_agent
    
    def simulate(self):
        for t in range(1000): # iterating over timesteps
            print('\n\nTimestep ', t)
            stock = dict()
            used_stock = dict()
            for food_category in self.foods:
                stock[food_category] = random.sample(range(0,15), len(self.foods[food_category])) # generates stock array of size n
                used_stock[food_category] = [0 for i in range(len(self.foods[food_category]))]
            original_stock = deepcopy(stock)
            print('Current Stock: ',stock)
            self.users.sort(key = lambda x: x.reward, reverse = True)  # sorting by how much user has adjusted so far
            for user in self.users:
                for food_category in self.foods:
                    request = user.request(food_category)
                    print('\nUser Request: ',self.foods[food_category][request])
                    if stock[food_category][request] > 2:
                        # allocate
                        stock[food_category][request] -= 1
                        used_stock[food_category][request] += 1
                        print('In Stock - Request Accepted')
                    elif stock[food_category][request] > 0: # low stock
                        # suggest, using the bandit for this user
                        food_agent = self.user_agent_map[user]
                        suggested_option = food_agent.suggest(food_category, stock[food_category])
                        print(self.foods[food_category][request], 'is low in stock. Would you be willing to replace with', self.foods[food_category][suggested_option],'?')
                        print('User Vegetarian:',user.vegetarian)
                        print('User Diabetic:', user.diabetic)
                        user_response = user.get_response(food_category, suggested_option)
                        food_agent.learn(food_category, suggested_option, user_response)
                        if user_response >= user.threshold:
                            print('User ACCEPTED')
                            feedback = random.choices(self.good_feedback)[0]
                            # user accepted
                            stock[food_category][suggested_option] -= 1
                            used_stock[food_category][suggested_option] += 1
                        else:
                            print('User REJECTED')
                            feedback = random.choices(self.bad_feedback)[0]
                            stock[food_category][request] -= 1
                            used_stock[food_category][request] += 1
                        print(feedback)
                        user.get_nudged(feedback)
                    else:
                        food_agent = self.user_agent_map[user]
                        suggested_option = food_agent.suggest(food_category, stock[food_category])
                        print(self.foods[food_category][request], 'is out of stock. Would you be willing to replace with', self.foods[food_category][suggested_option],'?')
                        print('User Vegetarian:',user.vegetarian)
                        print('User Diabetic:', user.diabetic)
                        user_response = user.get_response(food_category, suggested_option)
                        food_agent.learn(food_category, suggested_option, user_response)
                        if user_response >= user.threshold:
                            print('User ACCEPTED')
                            feedback = self.good_feedback[2]
                            # user accepted
                            stock[food_category][suggested_option] -= 1
                            used_stock[food_category][suggested_option] += 1
                        else:
                            print('User REJECTED')
                            feedback = self.bad_feedback[2]
                        print(feedback)
                        user.get_nudged(feedback)
            # print how much food used and how much wasted
            print('\nOriginal Stock: ',original_stock)  
            print('Stock Used: ', used_stock)
        # now print preferences we've calculated for each user
        print(self.foods)
        for i,user in enumerate(self.users):
            print('\nUser ',i)
            print('User Vegetarian: ', user.vegetarian)
            print('User Diabetic: ', user.diabetic)
            print('\nActual User Preferences:')
            for food_category in self.foods:
                print(food_category)
                print([self.foods[food_category][j] for j in user.preferences[food_category]])
            calculated_prefs = self.user_agent_map[user].get_learnt_preferences()
            print('\nCalculated User Preferences:')
            for food_category in self.foods:
                print(food_category)
                print(calculated_prefs[food_category])
            print('\nUser Reward: ', user.reward)
            print('User Adjusted', user.adjustments, 'Times Out Of', user.replacements)
            print(user.threshold)
            print('\nUser nudged by', round((((user.original_threshold-user.threshold)/user.original_threshold) * 100),2),'%.')


def main():
    allocator = FoodAllocator(20) # initialize allocator by setting number of users in simulation
    allocator.simulate()

if __name__== "__main__":
    main()


    

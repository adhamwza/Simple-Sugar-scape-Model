from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
import random
import matplotlib.pyplot as plt

class WealthAgent(Agent):
    def __init__(self, unique_id, model, initial_wealth):
        super().__init__(unique_id, model)
        self.wealth = initial_wealth

    def move(self):
        if self.unique_id == 'rich':
            x, y = self.pos
            neighbors = self.model.grid.get_neighbors((x, y), moore=True, include_center=False)
            if neighbors:
                other_agent = random.choice(neighbors)
                self.wealth += other_agent.wealth
                other_agent.wealth = 0

    def step(self):
        self.move()

class WealthModel(Model):
    def __init__(self, width, height, rich_agent_wealth):
        self.num_agents = 10
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)

        # Create agents
        for i in range(self.num_agents):
            agent = WealthAgent(i, self, 1)
            x = random.randrange(self.grid.width)
            y = random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))
            self.schedule.add(agent)

        # Create a rich agent with higher initial wealth
        rich_agent = WealthAgent('rich', self, rich_agent_wealth)
        x = random.randrange(self.grid.width)
        y = random.randrange(self.grid.height)
        self.grid.place_agent(rich_agent, (x, y))
        self.schedule.add(rich_agent)

        self.datacollector = DataCollector(agent_reporters={"Wealth": "wealth"})

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()

# Set up and run the model
width, height = 10, 10
rich_agent_wealth = 20
model = WealthModel(width, height, rich_agent_wealth)

for i in range(100):
    model.step()

# Collect data for analysis
data = model.datacollector.get_agent_vars_dataframe().reset_index()

# Print total wealth
total_wealth = data.groupby('Step')['Wealth'].sum()
print("Total Wealth Over Time:")
print(total_wealth)

# Scatter plot of final wealth distribution
plt.figure(figsize=(10, 5))
for agent in model.schedule.agents:
    x, y = agent.pos
    color = 'red' if agent.unique_id == 'rich' else 'blue'
    size = agent.wealth + 1  # Adjust size based on wealth
    plt.scatter(x, y, color=color, s=size)

plt.title('Wealth Distribution')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

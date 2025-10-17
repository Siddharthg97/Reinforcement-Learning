RL libraries are - 
1) stable baseline3 -




2) gymnasium
https://medium.com/@paulswenson2/an-introduction-to-building-custom-reinforcement-learning-environment-using-openai-gym-d8a5e7cf07ea
OpenAI Gym is an open source Python module which allows developers, researchers and data scientists to build reinforcement learning (RL) environments using a
pre-defined framework. The primary motivation for using Gym instead of just base Python or some other programming language is designed to interact with other RL Python modules.
One such module is stablebaselines3, which allows you to quickly train RL models on these environments without having to write all the algorithms yourself.



## Implementation
gymnaisium library
https://www.gymlibrary.ml/

Define environment and agent.
1) Create your own environmemt - libraries are there & default env are there - CartPole-v1
   we can also use gymnasium and create our own environment.

   
3) Need to define state, action space, obervation space and reward action, reward policy
   
observation space -It defines the format, structure, and limits of the input state (the environment’s output at each step).

Environment	Example Observations	Observation Space Type
CartPole-v1	[position, velocity, pole_angle, pole_angular_velocity]	Box(shape=(4,))

The observation space = what information the agent receives to make decisions.

It defines all possible actions the agent can take in the environment at any given time.

The action space = set of all valid moves.

Examples
Environment	Example Actions	Action Space Type
CartPole-v1	Move left or right	Discrete(2)


Concept	Meaning	Example
Observation Space	What the agent receives from the environment (state info)	“Pole angle = 0.3°, velocity = 0.7”
Action Space	What the agent can do in response	“Move cart left” or “Move cart right”
class BasicEnv(Env):
    __init__(self):
        pass
    
    step(self, action):
        pass
    reset(self):
        pass
    render(self):
        pass

DummyVecEnv stands for Dummy Vectorized Environment.
It’s a wrapper around your environment(s) that makes them compatible with algorithms in Stable-Baselines3 (SB3) — like PPO, A2C, DQN, etc.
Most RL algorithms in SB3 expect to receive input/output in a vectorized (batched) form — that is, they assume multiple environments running in parallel to collect experience faster.

However, if you only have a single environment, you still need to wrap it to provide a consistent interface.
That’s what DummyVecEnv does — it pretends (dummy) to be vectorized.

Converts the environment’s obs, reward, done, info into batch (vector) format:

Observations → shape (num_envs, obs_dim)

Rewards → shape (num_envs,)

But since it’s dummy, it just uses 1 environment and wraps data into an extra dimension.



## model training 
You’re creating an instance of your custom callback class (inheriting from BaseCallback).

check_freq=2000 means: inside your callback’s _on_step() method, it will print/log something every 2000 environment steps.

The callback is the “observer” that tracks training progress, logs info, or triggers custom behavior (saving model, stopping early, etc).


callback = StepLoggerCallback(check_freq=2000)

You’re creating an instance of your custom callback class (inheriting from BaseCallback).

check_freq=2000 means: inside your callback’s _on_step() method, it will print/log something every 2000 environment steps.

The callback is the “observer” that tracks training progress, logs info, or triggers custom behavior (saving model, stopping early, etc).



model.learn(total_timesteps=2000, callback=callback)

This starts training your PPO (or other algorithm) for 2000 timesteps.

During training, SB3 internally calls your callback methods:

_on_training_start() — once at the start

_on_step() — after each environment step

_on_rollout_end() — after each rollout

_on_training_end() — once at the end

Your StepLoggerCallback’s _on_step() method is invoked repeatedly, so if self.n_calls % 2000 == 0, it will log progress.

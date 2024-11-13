#import commit0
from agent.run_agent import run_agent as agent_module
import argparse
import os
import subprocess
from dotenv import load_dotenv
from custom_commit0 import custom_run_agent_for_repo

def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--split", type=str, default='parsel', help="The split of commit0 to implement and run unit test cases with.")
  #parser.add_argument("--persistence", type=int, default=1)
  parser.add_argument("--branch", default="commit0-test")
  parser.add_argument("--agent-name", default="aider")
  parser.add_argument("--model-name", default="gpt-4o-mini", help="Model to use with commit0 agent.")

  return parser.parse_args()

def agent_run(args):
  # Monkey patch, run_agent_for_repo to use our custom agent
  agent_module.run_agent_for_repo = custom_run_agent_for_repo

  # Now, when you call run_agent, it will use your custom run_agent_for_repo
  #agent_module.run_agent()
  # not sure if this will work!
  subprocess.run(["agent", "run", args.branch])

if __name__ == "__main__":
  args = get_args()
  
  load_dotenv() # for the API key

  ## SETUP commit0 & agent
  subprocess.run(["commit0", "setup", args.split])
  subprocess.run(["agent", "config", "--model-name", args.model_name, args.agent_name])
  
  # try running the code!
  agent_run(args)




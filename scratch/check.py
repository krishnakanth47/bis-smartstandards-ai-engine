import pickle
import json
import re

try:
    with open("data/index.meta", "rb") as f:
        meta = pickle.load(f)
        chunks = meta["chunks"]
        
        targets = ["IS 269", "IS 459", "IS 455", "IS 1489"]
        
        for target in targets:
            found = False
            for chunk in chunks:
                if target in chunk["text"] or target in chunk.get("standard_id", ""):
                    print(f"FOUND {target}!")
                    print(chunk["text"][:200])
                    found = True
                    break
            if not found:
                print(f"MISSING {target}!")
except Exception as e:
    print(f"Error: {e}")

import pickle
import sys

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            if module.startswith('numpy'):
                print(f"Found numpy reference: {module}.{name}")
            return super().find_class(module, name)
        except:
            print(f"Error loading: {module}.{name}")
            return None

with open('model.pkl', 'rb') as f:
    try:
        CustomUnpickler(f).load()
    except Exception as e:
        print(f"Exception: {e}")
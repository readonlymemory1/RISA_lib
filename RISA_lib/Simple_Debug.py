class Debug():
    def __init__(self, value, value_name=""):
        self.value = value
        self.value_name = value_name    
    def value_checker(self):
        if self.value_name=="":
            for v in range(len(self.value)):
                print(str(v)+': '+str(self.value[v]))
        else:
            for v in range(len(self.value)):
                print(str(self.value_name[v])+' : '+str(self.value[v]))
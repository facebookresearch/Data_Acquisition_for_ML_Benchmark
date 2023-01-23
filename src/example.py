from dam import Dam
print("Loading Dataset...")
instance=2 # instance id, can be 0,1,2,3,4
MyDam = Dam(instance=instance)
print("Dataset loaded!")
budget = MyDam.getbudget() # get budget
print("budget is:",budget)
# 3. Display seller_data 
buyer_data = MyDam.getbuyerdata() # get buyer data
print("buyer data is:",buyer_data)
mlmodel = MyDam.getmlmodel() # get ml model
print("mlmodel is",mlmodel)
sellers_id = MyDam.getsellerid() # seller ids
print("seller ids are", sellers_id)
for i in sellers_id:
    seller_i_price, seller_i_summary, seller_i_samples = MyDam.getsellerinfo(seller_id=int(i))
    print("seller ", i, " price: ", seller_i_price)
    print("seller ", i, " summary: ", seller_i_summary)
    print("seller ", i, " samples: ", seller_i_samples)
	
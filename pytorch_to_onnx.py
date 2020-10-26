import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
model = Net().to(device)



summary(model, (1, 28, 28))


batch_size = 1  #批处理大小
input_shape = (1, 28, 28)   #输入数据,改成自己的输入shape

# #set the model to inference mode
model.eval()

data = torch.zeros((1,1,28,28)).to(device)
with torch.no_grad():
    result = model.forward(data)
    print("print dummy result")
    print(result)

# device = 'cpu'
x = torch.randn(batch_size, *input_shape)   # 生成张量
x = x.to('cpu')
model.to('cpu').eval()
export_onnx_file = "test.onnx"		# 目的ONNX文件名
# torch.onnx.export(model,x,
#                   export_onnx_file,
#                   opset_version=10,
#                   do_constant_folding=True,	# 是否执行常量折叠优化
#                   input_names=["input"],	# 输入名
#                   output_names=["output"],	# 输出名
#                   dynamic_axes={"input":{0:"batch_size"},  # 批处理变量
#                                 "output":{0:"batch_size"}})

torch.onnx.export(model,x,
                  export_onnx_file,
                  opset_version=10,
                  do_constant_folding=True,	# 是否执行常量折叠优化
                  input_names=["input"],	# 输入名
                  output_names=["output"])	# 输出名

# torch.onnx.export(model,x,
#                 export_onnx_file,
#                 opset_version=10,
#                 do_constant_folding=True)
                
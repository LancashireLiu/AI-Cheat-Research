"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

data_loading.py

(0) MinMaxScaler: Min Max normalizer
(1) sine_data_generation: Generate sine dataset
(2) real_data_loading: Load and preprocess real data
  - stock_data: https://finance.yahoo.com/quote/GOOG/history?p=GOOG
  - energy_data: http://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction
"""

## Necessary Packages
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def MinMaxScaler(data):
  """Min Max normalizer.
  
  Args:
    - data: original data
  
  Returns:
    - norm_data: normalized data
  """
  numerator = data - np.min(data, 0)
  denominator = np.max(data, 0) - np.min(data, 0)
  norm_data = numerator / (denominator + 1e-7)
  return norm_data


def sine_data_generation (no, seq_len, dim):
  """Sine data generation.
  
  Args:
    - no: the number of samples
    - seq_len: sequence length of the time-series
    - dim: feature dimensions
    
  Returns:
    - data: generated data
  """  
  # Initialize the output
  data = list()

  # Generate sine data
  for i in range(no):      
    # Initialize each time-series
    temp = list()
    # For each feature
    for k in range(dim):
      # Randomly drawn frequency and phase
      freq = np.random.uniform(0, 0.1)            
      phase = np.random.uniform(0, 0.1)
          
      # Generate sine signal based on the drawn frequency and phase
      temp_data = [np.sin(freq * j + phase) for j in range(seq_len)] 
      temp.append(temp_data)
        
    # Align row/column
    temp = np.transpose(np.asarray(temp))        
    # Normalize to [0,1]
    temp = (temp + 1)*0.5
    # Stack the generated data
    data.append(temp)
                
  return data
    
'''def real_data_loading (data_name, seq_len):
  """Load and preprocess real-world datasets.
  
  Args:
    - data_name: stock or energy
    - seq_len: sequence length
    
  Returns:
    - data: preprocessed data.
  """  
  assert data_name in ['stock','energy']
  
  if data_name == 'stock':
    ori_data = np.loadtxt('data/stock_data.csv', delimiter = ",",skiprows = 1)
  elif data_name == 'energy':
    ori_data = np.loadtxt('data/energy_data.csv', delimiter = ",",skiprows = 1)
        
  # Flip the data to make chronological data
  ori_data = ori_data[::-1]
  # Normalize the data
  ori_data = MinMaxScaler(ori_data)
    
  # Preprocess the dataset
  temp_data = []    
  # Cut data by sequence length
  for i in range(0, len(ori_data) - seq_len):
    _x = ori_data[i:i + seq_len]
    temp_data.append(_x)
        
  # Mix the datasets (to make it similar to i.i.d)
  idx = np.random.permutation(len(temp_data))    
  data = []
  for i in range(len(temp_data)):
    data.append(temp_data[idx[i]])
    
  return data'''

#zhuanhuanzuobiaozhou zhushi
'''def real_data_loading(data_name, seq_len):
  """Load and preprocess mouse movement datasets.

  Args:
      - data_name: stock or energy
      - seq_len: sequence length (in milliseconds, e.g., 1000ms)

  Returns:
      - data: preprocessed data.
  """
  assert data_name in ['stock', 'energy']

  if data_name == 'stock':
    file_path = 'E:/反外挂文件/GAN/TimeGAN/data/stock.csv'
  elif data_name == 'energy':
    file_path = 'E:/反外挂文件/GAN/TimeGAN/data/energy.csv'
  else:
    raise ValueError("Unsupported data name")

  # Load the mouse movement data
  ori_data = np.loadtxt(file_path, delimiter=",", skiprows=1)

  # Normalize the data
  ori_data = MinMaxScaler(ori_data)

  # Preprocess the dataset
  temp_data = []
  step = 100  # Step size (100 ms), adjust as needed
  # Cut data by sequence length with the given step
  for i in range(0, len(ori_data) - seq_len + 1, step):
    _x = ori_data[i:i + seq_len]
    temp_data.append(_x)

  # Mix the datasets (to make it similar to i.i.d)
  idx = np.random.permutation(len(temp_data))
  data = []
  for i in range(len(temp_data)):
    data.append(temp_data[idx[i]])
'''
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def euclidean_distance(point1, point2):
  return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def rotate_point(point, angle):
  x, y = point
  cos_theta = np.cos(angle)
  sin_theta = np.sin(angle)
  x_new = x * cos_theta + y * sin_theta
  y_new = -x * sin_theta + y * cos_theta
  return np.array([x_new, y_new])


def transform_coordinates(window):
  start_point = window[0]
  end_point = window[-1]
  d = euclidean_distance(start_point, end_point)

  # 计算旋转角度
  delta_x = end_point[0] - start_point[0]
  delta_y = end_point[1] - start_point[1]
  angle = np.arctan2(delta_y, delta_x)

  transformed_window = []
  for point in window:
    relative_point = point - start_point
    transformed_point = rotate_point(relative_point, -angle)
    transformed_window.append(transformed_point)

  transformed_window = np.array(transformed_window)

  # 将坐标转换为整数
  transformed_window = np.rint(transformed_window).astype(int)

  return transformed_window


def sliding_window_transform(ori_data, window_size=1000, step_size=100):
  transformed_data = []
  for start_idx in range(0, len(ori_data) - window_size + 1, step_size):
    window = ori_data[start_idx:start_idx + window_size]
    transformed_window = transform_coordinates(window)
    transformed_data.append(transformed_window)

  return np.array(transformed_data)


def real_data_loading(data_name, seq_len):
  """Load and preprocess stock or energy datasets.

  Args:
      - data_name: stock or energy
      - seq_len: sequence length (in number of data points, e.g., 1000)

  Returns:
      - data: preprocessed data.
  """
  assert data_name in ['stock', 'energy']

  if data_name == 'stock':
    file_path = 'C:/Users/Admin/Documents/WeChat Files/wxid_epckp5jbazzj22/FileStorage/File/2024-05/TimeGAN/TimeGAN/data/stock.csv'
  elif data_name == 'energy':
    file_path = 'C:/Users/Admin/Documents/WeChat Files/wxid_epckp5jbazzj22/FileStorage/File/2024-05/TimeGAN/TimeGAN/data/energy.csv'
  else:
    raise ValueError("Unsupported data name")

  # Load the mouse movement data
  ori_data = np.loadtxt(file_path, delimiter=",", skiprows=1)

  # Print the first few rows of ori_data to ensure it's loaded correctly
  print("Original data (first 5 rows):")
  print(ori_data[:5])

  # Apply sliding window and coordinate transformation
  transformed_data = sliding_window_transform(ori_data, window_size=seq_len, step_size=100)

  # Normalize the data
  scaler = MinMaxScaler()
  transformed_data = transformed_data.reshape(-1, transformed_data.shape[-1])
  transformed_data = scaler.fit_transform(transformed_data).reshape(-1, seq_len, transformed_data.shape[-1])

  # Print the first few rows of normalized data to ensure it's normalized correctly
  print("Processed data (first 5 rows):")
  print(transformed_data[:5])

  # Save preprocessed data
  np.save('preprocessed_data.npy', transformed_data)
  flat_data = transformed_data.reshape(-1, transformed_data.shape[-1])
  pd.DataFrame(flat_data).to_csv('preprocessed_data.csv', index=False)

  return transformed_data






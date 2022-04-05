# Multi-output Model Writeup
> A writeup of multi-output model architecture with TensorFlow utilizing the functional API.

## General Architecture 

Using the TensorFlow's Functional API, one can create branched models where a single layer can be referenced in two or more following layers in the architecture. This of course can be used to indicate separate output layers.

```python
input_layer = Input(shape=(28, 28))
first_dense = Dense(units='64', activation='relu')(input_layer)
second_dense = Dense(units='64', activation='relu')(first_dense)

output_1 = Dense(units='1', name='output_1')(second_dense)
third_dense = Dense(units='64', activation='relu')(second_dense)

output_2 = Dense(units='1', name='output_2')(third_dense)
```

As it can be observed in the above code the `second_dense` layer is referenced twice, causing a branch intermediate an output and a continuing dense layer.

## Real Applications of The Multi-output Model Architecture.
|**Title**|**Description**|**File**|
|---------|---------------|--------|
|Energy Efficiency Regression|Excel dataset found in the UCI Machine Learning Repository.||

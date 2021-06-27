def get_model():
    model_vgg16_conv = VGG16(weights='imagenet', include_top=False)

    # Dong bang cac layer
    for layer in model_vgg16_conv.layers:
        layer.trainable = False

    # Tao model
    input = Input(shape=(128, 128, 3), name='image_input') #input đầu vào là các bức ảnh kích thước 128x128x3
    output_vgg16_conv = model_vgg16_conv(input) # lấy output của VGG16 và làm input của các layers FC thêm vào

    # Them cac layer FC va Dropout
    x = Flatten(name='flatten')(output_vgg16_conv) # dàn phẳng các bức ảnh thành mảng 1 chiều
    x = Dense(4096, activation='relu', name='fc1')(x)#lớp dense
    x = Dropout(0.5)(x)#giảm việc bị overfiting
    x = Dense(4096, activation='relu', name='fc2')(x)# các lớp fully connected
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu', name='fc3')(x)
    x = Dropout(0.5)(x)
    x = Dense(5, activation='softmax', name='predictions')(x)#Softmax chuyển đổi một vectơ giá trị thành phân phối xác suất.
#Các phần tử của vectơ đầu ra nằm trong phạm vi (0, 1) và tổng bằng 1.
    # Compile
    my_model = Model(inputs=input, outputs=x)#Create your own model (tạo model của mình với input dau vao là ảnh dau ra là du doan trai cay loai nao)
    my_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #định cấu hình để đào tạo với loss(hàm mất mát,Tính toán tổn thất chéo giữa các nhãn và dự đoán do sử dụng nhiều lớp label(label dạng one-hot),Nếu bạn muốn cung cấp nhãn bằng cách sử dụng one-hotđại diện, vui lòng sử dụng CategoricalCrossentropymất má
    #Tối ưu hóa Adam là một phương pháp giảm độ dốc ngẫu nhiên dựa trên ước tính thích ứng của các khoảnh khắc bậc nhất và bậc hai.
    #đánh giá hiệu suất của mô hình của bạn(metrics )
    return my_model
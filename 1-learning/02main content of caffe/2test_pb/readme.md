How to use these files?

1.For file yll.helloworld.proto,you need run the command of this.And it will generate two files named yll.helloworld.pb.cc  and yll.helloworld.pb.h

`$ protoc -I=. --cpp_out=. yll.helloworld.proto`

2.Then for file test_pb.cpp,you need to run the commamd of this.It will generate an executable file named test_pb.

`$ gcc -o test_pb test_pb.cpp yll.helloworld.pb.cc -lprotobuf -lstdc++`

3.At last,you could run this by

`$ ./test_pb`
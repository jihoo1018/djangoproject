from django.http import JsonResponse
from rest_framework.authtoken.models import Token
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import JSONParser
from rest_framework.response import Response
from blog.blog_users.models import BlogUser
from blog.blog_users.serializer import BlogUserSerializer
from blog.blog_users.services import UserService


'''@api_view(['GET'])
@parser_classes([JSONParser])
def login(request):
    users = UserService().create_users()
    return JsonResponse({'users ': users})'''


@api_view(['GET'])
@parser_classes([JSONParser])
def user_list(request):
    if request.method == 'GET':
        serializer = BlogUserSerializer(BlogUser.objects.all(), many=True)
        return Response(serializer.data)


@api_view(['POST'])
@parser_classes([JSONParser])
def loginform(request):
    print(" 진입 ## ")
    try:
        print(f"로그인 정보 : {request.data}")
        info = request.data
        loginuser = BlogUser.objects.get(email=info['email'])
        print(f"해당 email을 가진 Userid:{loginuser.blog_userid}")
        print(f"해당 email을 가진 password:{loginuser.password}")
        print(f"해당 email을 가진 info password :{info['password']}")
        if loginuser.password == info["password"]:
            dbuser = BlogUser.objects.all().filter(blog_userid=loginuser.blog_userid).values()[0]
            print(f"dbuser is {dbuser}")
            serialize = BlogUserSerializer(dbuser, many=False)
            return JsonResponse(data=serialize.data, safe=False)
        # dictionary이외를 받을 경우, 두번째 argument를 safe=False로 설정해야한다.
        else:
            return Response("비번이 틀립니다")
    except:
        return Response("이메일이 틀립니다")

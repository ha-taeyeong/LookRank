from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch

# 1. 장치 설정 (GPU 있으면 GPU 사용)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. 얼굴 검출기(MTCNN)와 임베딩 모델(InceptionResnetV1) 불러오기
mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# 3. 테스트할 이미지 불러오기 (파일 경로 수정)
img = Image.open('test_face.jpg')  # 예: 같은 폴더에 있는 얼굴 사진

# 4. 얼굴 검출
face = mtcnn(img)
if face is not None:
    face = face.unsqueeze(0).to(device)   # 배치 차원 추가
    # 5. 얼굴 임베딩 추출
    embedding = resnet(face)
    print("임베딩 벡터 크기:", embedding.shape)
    print("임베딩 벡터 예시:", embedding[0][:5])  # 앞부분 5개만 출력
else:
    print("얼굴을 찾지 못했습니다.")

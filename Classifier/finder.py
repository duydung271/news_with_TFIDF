import pickle
from tools.preprocess import *
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.sparsefuncs import mean_variance_axis

#load vectorizer
f = open('Classifier/vectorizer.cache', 'rb')
vectorizer = pickle.load(f)
f.close()
stopwords = get_stopwords()

#load dataset
dataset = get_data()
clean_data = [clean_text(choice_content(item), stopwords) for item in dataset ]


rep_dataset = vectorizer.transform(clean_data)


queries = [{
        'title': 'Cạnh tranh gắt gao về điểm số tại chung kết AI Hackathon 2022',
        'description': 'Sáu đội thi bám đuổi điểm số trong 180 phút giải 5 bộ dữ liệu về lịch trình du lịch, tại vòng chung kết sáng nay (10/9)',
        'content': '''
            Quy Nhơn AI Hackathon 2022 vừa diễn ra trận chung kết vào 9h ngày 10/9 tại Đại học FPT Quy Nhơn.
            Tham gia tranh tài là 6 đội: Laugh Tale, AIC, PSPCapital, CTA Matrix, CTA Gà con và NamCyan.
            Vòng chung kết là bài toán ứng dụng AI trong việc lên lịch trình thích hợp nhất để du lịch Quy Nhơn (Bình Định).
            Ban tổ chức công bố bộ đề gồm 5 tập dữ liệu để các nhóm đánh giá hiệu năng của thuật toán.
            Dựa trên thông tin của từng du khách như độ tuổi, ngân sách, thời gian...; 
            6 đội thi sẽ tối ưu thuật toán dựa trên model mẫu để đưa ra một lịch trình di chuyển qua các địa điểm du lịch, gợi ý các nhà hàng hay khách sạn phù hợp.
            Tổng hợp điểm của 5 đề bài, đội có chiến thuật và thuật toán tối ưu nhất sẽ giành chiến thắng.
        '''
    },
    {
        'title':'18 dự án vào chung kết AI Awards 2022',
        'description': 'Giải AI Awards do VnExpress tổ chức đã chọn được 18 dự án vào vòng chung kết để tìm ra sản phẩm xuất sắc trao giải chung cuộc. ',
        'content': '''
            Giải thưởng bình chọn Sản phẩm ứng dụng Trí tuệ nhân tạo 2022 (AI Awards 2022) chia thành 2 hệ thống giải gồm: AI Awards và AI Tech Matching.
            Ban tổ chức và Hội đồng giám khảo đã chọn ra hai danh sách giải pháp và sản phẩm trong số 38 bài thi vượt qua vòng sơ loại, phù hợp với tiêu chí cho từng hệ thống giải trên
            Ở bảng AI Awards, trong số 18 bài thi Hội đồng giám khảo sẽ chọn ra top 5 dự án xuất sắc và trao giải. Mỗi dự án sẽ nhận thưởng 150 triệu đồng, trong đó 30 triệu đồng tiền mặt và gói truyền thông trị giá 120 triệu đồng trên VnExpress.
            Hạng mục này vinh danh những sản phẩm, giải pháp ứng dụng AI nổi bật ứng dụng trong sản xuất doanh nghiệp và cuộc sống.
            Các dự án phải hướng tới thay đổi cuộc sống con người từ mức độ cơ bản đến toàn diện;
            có tính sáng tạo độc đáo trong việc ứng dụng AI.
        '''
    },{
        'title': '13 dự án vào vòng AI Tech Matching',
        'description': 'Đây là các dự án tham gia cuộc thi AI Awards do VnExpress tổ chức, được chọn vào vòng kết nối với doanh nghiệp nhằm hoàn thiện giải pháp, sản phẩm, đưa ra thị trường.',
        'content': '''
            Giải thưởng bình chọn Sản phẩm ứng dụng Trí tuệ nhân tạo 2022 (AI Awards 2022) năm nay chia thành 2 hệ thống giải gồm: AI Awards và AI Tech Matching.
            Ban tổ chức và Hội đồng giám khảo đã chọn ra hai danh sách giải pháp và sản phẩm phù hợp với tiêu chí cho từng hệ thống giải trên.
            Ở hạng mục AI Tech Matching, nhà tài trợ Aus4innovation sẽ đầu tư tổng chi phí 60.000 AUD (gần một tỷ đồng) nhằm hỗ trợ các sản phẩm AI tiềm năng phát triển và ứng dụng trong thực tế, có cơ hội kết nối với doanh nghiệp nhằm hoàn thiện giải pháp, sản phẩm, đưa ra thị trường.
            Các dự án phải đáp ứng các tiêu chí như sản phẩm giải quyết được vấn đề cụ thể của khách hàng,
            nhu cầu đang tồn tại thực tế trên thị trường Việt Nam. 
            Mô hình kinh doanh giải quyết hiệu quả vấn đề khách hàng và quy mô thị trường tiềm năng đủ lớn tại Việt Nam, 
            trong đó tiềm năng xuất khẩu là một lợi thế. Sản phẩm có tính cạnh tranh, thu hút được khách hàng tiềm năng và bền vững.
        '''
    },{
        'title': 'Ngày hội trí tuệ nhân tạo Việt Nam 2022 sẽ diễn ra 22-23/9',
        'description': 'Chuyên gia trong nước và quốc tế, đại diện doanh nghiệp và nhà quản lý sẽ cùng thảo luận các giải pháp đưa AI vào cuộc sống, sản xuất trong Ngày hội Trí tuệ nhân tạo Việt Nam sắp tới.',
        'content': '''
        Ngày hội Trí tuệ nhân tạo Việt Nam (AI4VN) sẽ diễn ra trong hai ngày (22-23/9) tại Khách sạn Grand Plaza,
        Hà Nội. Với chủ đề "AI phục hồi kinh tế, định hình tương lai", sự kiện là bức tranh toàn cảnh từ góc nhìn ứng dụng,
        sự hưởng ứng của doanh nghiệp trong hệ sinh thái phát triển AI tại Việt Nam.
        Chương trình bao gồm 3 workshop trước thềm phiên toàn thể, với các chủ đề: Giải pháp AI trong lĩnh vực tài chính - ngân hàng;
        Kết nối nguồn nhân lực và Tự động hóa trong sản xuất.
        Tại mỗi hội thảo, các diễn giả đến từ doanh nghiệp, viện nghiên cứu,
        trường đại học sẽ chia sẻ những câu chuyện thực tế trong phát triển,
        ứng dụng công nghệ cũng như việc chuẩn bị nguồn nhân lực trong lĩnh vực này.
        Phiên chính diễn ra sáng 23/9 sẽ có sự tham gia của lãnh đạo chính phủ, bộ, ngành, chuyên gia, doanh nghiệp thực chiến trong nước và quốc tế về lĩnh vực AI.
        '''
    }
]

clean_queries = [clean_text(choice_content(query), stopwords) for query in queries]
rep_queries = vectorizer.transform(clean_queries)

rep_query, _ = mean_variance_axis(rep_queries, axis=0)
rep_query = rep_query.reshape(1,-1)

k = 20
neigh = NearestNeighbors(n_neighbors=k)
neigh.fit(rep_dataset)

scores, indexes = neigh.kneighbors(rep_query)

for i in range(len(indexes)):
    for stt in range(k):
        print(f'-----------------------------------------------------------------------')
        print(f'score {scores[i][stt]}')
        print(f'artical: {dataset[indexes[i][stt]]}')






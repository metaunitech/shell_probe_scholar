import fitz, io, os
from PIL import Image
from loguru import logger
import re
from modules.models.pdf_extract import *


class Paper:
    def __init__(self, path, title=None, url=None, abs=None, authers=None, update_ts=None, published_ts=None):
        # 初始化函数，根据pdf路径初始化Paper对象
        self.url = url if url else ''  # 文章链接
        self.path = path  # pdf路径
        self.section_names = []  # 段落标题
        self.section_texts = {}  # 段落内容
        self.abs = abs if abs else ""
        self.title_page = 0
        if not title:
            self.pdf = fitz.open(self.path)  # pdf文档
            self.title = self.get_title()
            self.parse_pdf()
        else:
            self.title = title
            self.parse_pdf()
        if not self.title or len(self.title) >= 100:
            self.title = '.'.join(list(os.path.basename(path).split('.'))[:-1])
        self.authers = authers if authers else []
        self.roman_num = ["I", "II", 'III', "IV", "V", "VI", "VII", "VIII", "IIX", "IX", "X"]
        self.digit_num = [str(d + 1) for d in range(10)]
        self.first_image = ''
        self.pdf.close()
        self.PDFF2data = parsePDF_PDFFigures2(path)
        self.pdf = fitz.open(self.path)
        self.update_ts = update_ts
        self.published_ts = published_ts

    def clean_up(self):
        logger.info("PDF CLOSED")
        self.pdf.close()

    def parse_pdf(self):
        self.pdf = fitz.open(self.path)  # pdf文档
        if self.pdf.page_count == 0:
            logger.error(f"File is broken: {self.title}")
            self.clean_up()
            raise Exception("File broken.")
        self.text_list = [page.get_text() for page in self.pdf]
        self.all_text = ' '.join(self.text_list)
        self.section_page_dict = self._get_all_page_index()  # 段落与页码的对应字典
        logger.debug(f"section_page_dict: {self.section_page_dict}")
        self.section_text_dict = self._get_all_page()  # 段落与内容的对应字典
        self.section_text_dict.update({"title": self.title})
        self.section_text_dict.update({"paper_info": self.get_paper_info()})
        self.reference_list = self.get_reference()


    def get_paper_info(self):
        first_page_text = self.pdf[self.title_page].get_text()
        if "Abstract" in self.section_text_dict.keys():
            abstract_text = self.section_text_dict['Abstract']
        else:
            abstract_text = self.abs
        first_page_text = first_page_text.replace(abstract_text, "")
        return first_page_text

    def get_image_path(self, image_path=''):
        """
        将PDF中的第一张图保存到image.png里面，存到本地目录，返回文件名称，供gitee读取
        :param filename: 图片所在路径，"C:\\Users\\Administrator\\Desktop\\nwd.pdf"
        :param image_path: 图片提取后的保存路径
        :return:
        """
        import time
        # open file
        self.pdf = fitz.open(self.path)  # pdf文档
        max_size = 0
        with fitz.Document(self.path) as my_pdf_file:
            # 遍历所有页面
            imgs = {}
            for page_number in range(1, len(my_pdf_file) + 1):
                # 查看独立页面
                page = my_pdf_file[page_number - 1]
                # # 查看当前页所有图片
                # images = page.get_images()
                # 遍历当前页面所有图片
                for image_number, image in enumerate(page.get_images(), start=1):
                    # 访问图片xref
                    xref_value = image[0]
                    # image name
                    image_name = image[-2]
                    # 提取图片信息
                    base_image = my_pdf_file.extract_image(xref_value)
                    # 访问图片
                    image_bytes = base_image["image"]
                    # 获取图片扩展名
                    ext = base_image["ext"]
                    # 加载图片
                    image = Image.open(io.BytesIO(image_bytes))
                    image_size = image.size[0] * image.size[1]
                    if image_size > max_size:
                        max_size = image_size
                    imgs[image_name] = image
        output = {}

        for image_name in imgs.keys():
            image_size = image.size[0] * image.size[1]
            image_file_name = f"image_{image_name}.{ext}"
            logger.info(image_name)
            im_path = os.path.join(image_path, image_file_name)
            logger.debug(f"im_path: {im_path}")

            max_pix = 480

            if image.size[0] > image.size[1]:
                min_pix = int(image.size[1] * (max_pix / image.size[0]))
                newsize = (max_pix, min_pix)
            else:
                min_pix = int(image.size[0] * (max_pix / image.size[1]))
                newsize = (min_pix, max_pix)
            image = image.resize(newsize)
            image.save(open(im_path, "wb"))
            output[image_name] = image_path


        return output

    # 定义一个函数，根据字体的大小，识别每个章节名称，并返回一个列表
    def get_chapter_names(self, ):
        # # 打开一个pdf文件
        doc = fitz.open(self.path)  # pdf文档
        text_list = [page.get_text() for page in doc]
        all_text = ''
        for text in text_list:
            all_text += text
        # # 创建一个空列表，用于存储章节名称
        chapter_names = []
        for line in all_text.split('\n'):
            if '.' in line:
                point_split_list = line.split('.')
                space_split_list = line.split(' ')
                if 1 < len(space_split_list) < 5:
                    if 1 < len(point_split_list) < 5 and (
                            point_split_list[0] + '\n' in self.roman_num or point_split_list[
                        0] + '\n' in self.digit_num):

                        logger.debug(f"line: {line}")
                        chapter_names.append(line)
                        # 这段代码可能会有新的bug，本意是为了消除"Introduction"的问题的！
                    elif 1 < len(point_split_list) < 5:
                        logger.debug(f"line: {line}")
                        chapter_names.append(line)

        return chapter_names

    def get_title(self):
        doc = self.pdf  # 打开pdf文件
        max_font_size = 0  # 初始化最大字体大小为0
        # max_string = ""  # 初始化最大字体大小对应的字符串为空
        max_font_sizes = [0]
        for page_index, page in enumerate(doc):  # 遍历每一页
            text = page.get_text("dict")  # 获取页面上的文本信息
            blocks = text["blocks"]  # 获取文本块列表
            for block in blocks:  # 遍历每个文本块
                if block["type"] == 0 and len(block['lines']):  # 如果是文字类型
                    if len(block["lines"][0]["spans"]):
                        font_size = block["lines"][0]["spans"][0]["size"]  # 获取第一行第一段文字的字体大小
                        max_font_sizes.append(font_size)
                        if font_size > max_font_size:  # 如果字体大小大于当前最大值
                            max_font_size = font_size  # 更新最大值
                            max_string = block["lines"][0]["spans"][0]["text"]  # 更新最大值对应的字符串
        max_font_sizes.sort()
        self.font_sizes_list = max_font_sizes
        logger.debug(f"max_font_sizes {max_font_sizes[-10:]}")
        cur_title = ''
        for page_index, page in enumerate(doc):  # 遍历每一页
            text = page.get_text("dict")  # 获取页面上的文本信息
            blocks = text["blocks"]  # 获取文本块列表
            for block in blocks:  # 遍历每个文本块
                if block["type"] == 0 and len(block['lines']):  # 如果是文字类型
                    for line in block['lines']:
                        if len(line["spans"]):
                            cur_string = line["spans"][0]["text"]  # 更新最大值对应的字符串
                            # font_flags = line["spans"][0]["flags"]  # 获取第一行第一段文字的字体特征
                            font_size = line["spans"][0]["size"]  # 获取第一行第一段文字的字体大小
                            # print(font_size)
                            if abs(font_size - max_font_sizes[-1]) < 0.3 or abs(font_size - max_font_sizes[-2]) < 0.3:
                                # print("The string is bold.", max_string, "font_size:", font_size, "font_flags:", font_flags)
                                if len(cur_string) > 4 and "arXiv" not in cur_string:
                                    # print("The string is bold.", max_string, "font_size:", font_size, "font_flags:", font_flags)
                                    if cur_title == '':
                                        cur_title += cur_string
                                    else:
                                        cur_title += ' ' + cur_string
                                self.title_page = page_index
        title = cur_title.replace('\n', ' ')
        return title

    @staticmethod
    def extract_citation_info(text):
        # 正则表达式提取引用
        citation_pattern = re.compile(
            r'\[([\s\S]+?)\]\s([\s\S]+?)(?:\s+In\s([\s\S]+?))?(?:\s[Aa]rXiv,\s([\s\S]+?),\s(\d{4}))?\.')

        matches = citation_pattern.findall(text)

        # 提取标题、作者和Arxiv号
        citations = []
        for match in matches:
            authors = match[0].split(', ')
            title = match[1]
            source = match[2] if match[2] else None
            arxiv_number = match[3] if match[3] else None
            year = match[4] if match[4] else None

            citation_info = {
                'authors': authors,
                'title': title,
                'source': source,
                'arxiv_number': arxiv_number,
                'year': year
            }

            citations.append(citation_info)

        return citations

    @staticmethod
    def extract_arxiv_paper(text):
        res = re.search(r'arXiv:\d+\.\d+')

    def get_reference(self):
        reference_page_idx = self.section_page_dict.get('References')
        if reference_page_idx is None:
            logger.error("Cannot find references.")
            return []
        logger.debug(f"Reference page range: {reference_page_idx}-{self.pdf.page_count - 1}")
        reference_pages_raw = ''
        for i in range(reference_page_idx, self.pdf.page_count):
            reference_pages_raw += self.pdf[i].get_text()
        # logger.debug(reference_pages_raw)

    def _get_all_page_index(self):
        # 定义需要寻找的章节名称列表
        section_list = ["Abstract",
                        'Introduction', 'Related Work', 'Background',
                        "Preliminary", "Problem Formulation",
                        'Methods', 'Methodology', "Method", 'Approach', 'Approaches',
                        # exp
                        "Materials and Methods", "Experiment Settings",
                        'Experiment', "Experimental Results", "Evaluation", "Experiments",
                        "Results", 'Findings', 'Data Analysis',
                        "Discussion", "Results and Discussion", "Conclusion",
                        'References']
        # 初始化一个字典来存储找到的章节和它们在文档中出现的页码
        section_page_dict = {}
        # 遍历每一页文档
        for page_index, page in enumerate(self.pdf):
            # 获取当前页面的文本内容
            cur_text = page.get_text()
            # 遍历需要寻找的章节名称列表
            for section_name in section_list:
                # 将章节名称转换成大写形式
                section_name_upper = section_name.upper()
                # 如果当前页面包含"Abstract"这个关键词
                if "Abstract" == section_name and section_name in cur_text:
                    # 将"Abstract"和它所在的页码加入字典中
                    section_page_dict[section_name] = page_index
                # 如果当前页面包含章节名称，则将章节名称和它所在的页码加入字典中
                else:
                    if section_name + '\n' in cur_text:
                        section_page_dict[section_name] = page_index
                    elif section_name_upper + '\n' in cur_text:
                        section_page_dict[section_name] = page_index
        # 返回所有找到的章节名称及它们在文档中出现的页码
        return section_page_dict

    def _get_all_page(self):
        """
        获取PDF文件中每个页面的文本信息，并将文本信息按照章节组织成字典返回。

        Returns:
            section_dict (dict): 每个章节的文本信息字典，key为章节名，value为章节文本。
        """
        text = ''
        text_list = []
        section_dict = {}

        # 再处理其他章节：
        text_list = [page.get_text() for page in self.pdf]
        for sec_index, sec_name in enumerate(self.section_page_dict):
            logger.debug(','.join([str(sec_index), sec_name, str(self.section_page_dict[sec_name])]))
            if sec_index <= 0 and self.abs:
                continue
            else:
                # 直接考虑后面的内容：
                start_page = self.section_page_dict[sec_name]
                if sec_index < len(list(self.section_page_dict.keys())) - 1:
                    end_page = self.section_page_dict[list(self.section_page_dict.keys())[sec_index + 1]]
                else:
                    end_page = len(text_list)
                logger.debug(f"start_page, end_page: {start_page}, {end_page}")
                cur_sec_text = ''
                if end_page - start_page == 0:
                    if sec_index < len(list(self.section_page_dict.keys())) - 1:
                        next_sec = list(self.section_page_dict.keys())[sec_index + 1]
                        if text_list[start_page].find(sec_name) == -1:
                            start_i = text_list[start_page].find(sec_name.upper())
                        else:
                            start_i = text_list[start_page].find(sec_name)
                        if text_list[start_page].find(next_sec) == -1:
                            end_i = text_list[start_page].find(next_sec.upper())
                        else:
                            end_i = text_list[start_page].find(next_sec)
                        cur_sec_text += text_list[start_page][start_i:end_i]
                else:
                    for page_i in range(start_page, end_page):
                        #                         print("page_i:", page_i)
                        if page_i == start_page:
                            if text_list[start_page].find(sec_name) == -1:
                                start_i = text_list[start_page].find(sec_name.upper())
                            else:
                                start_i = text_list[start_page].find(sec_name)
                            cur_sec_text += text_list[page_i][start_i:]
                        elif page_i < end_page:
                            cur_sec_text += text_list[page_i]
                        elif page_i == end_page:
                            if sec_index < len(list(self.section_page_dict.keys())) - 1:
                                next_sec = list(self.section_page_dict.keys())[sec_index + 1]
                                if text_list[start_page].find(next_sec) == -1:
                                    end_i = text_list[start_page].find(next_sec.upper())
                                else:
                                    end_i = text_list[start_page].find(next_sec)
                                cur_sec_text += text_list[page_i][:end_i]
                section_dict[sec_name] = cur_sec_text.replace('-\n', '').replace('\n', ' ')
        return section_dict

    ### MODIFIED FROM CHATPAPER2XMIND

    # def get_section_equationdict(self, legacy=False):
    #     """
    #     Get equation dict of each section
    #
    #     :param legacy: True for legacy equation extraction method
    #     :return: Dict of section titles with tuple item list
    #     (equation_text_pos, page_number, equation_bbox)
    #     """
    #     eqa_ls = []
    #     for i in range(len(self.pdf)):
    #         page = self.pdf[i]
    #         if legacy:
    #             eq_box = get_eqbox(page)
    #         else:
    #             eq_box = getEqRect(page)
    #         if eq_box:
    #             for box in eq_box:
    #                 eqa_ls.append((get_box_textpos(page, box, self.all_text),
    #                                i, box))
    #     section_title = self.get_section_titles()
    #     pos_dict = self.get_section_textposdict()
    #     section_dict = {}
    #     for title in section_title[:-1]:
    #         section_dict[title] = []
    #         for eqa in eqa_ls:
    #             if eqa[0] > pos_dict[title][0] and eqa[0] <= pos_dict[title][1]:
    #                 section_dict[title].append(eqa)
    #             elif section_dict[title]:
    #                 break
    #     return section_dict
    #
    # def gen_image(self, snap_with_caption, verbose=False):
    #     """
    #     Generate image for each section in xmind (Figure/Table)
    #     """
    #     section_names = self.paper.get_section_titles()
    #     img_dict = self.paper.get_section_imagedict(
    #         snap_with_caption=snap_with_caption, verbose=verbose)
    #     for name in section_names[:-1]:
    #         img_ls = img_dict.get(name)
    #         if img_ls:
    #             for img in img_ls:
    #                 img_tempdir = get_objpixmap(self.paper.pdf, img)
    #                 topic = topic_search(self, name).addSubTopicbyImage(
    #                     img_tempdir, img_ls.index(img))
    #                 # FIXME: This is a temporary solution for compatibility
    #                 if len(img) == 4:
    #                     topic.setTitle(img[3])
    #                     topic.setTitleSvgWidth()

    # def gen_equation(self, legacy=True):
    #     """
    #     Generate equation for each section in xmind
    #
    #     :param legacy: if True, use legacy method to extract\
    #         equation, else use new method.
    #     `NOTE` It seems that legacy method is more accurate.
    #     """
    #     section_names = self.paper.get_section_titles()
    #     eqa_dict = self.paper.get_section_equationdict(legacy=legacy)
    #     for name in section_names[:-1]:
    #         eqa_ls = eqa_dict.get(name)
    #         if eqa_ls:
    #             for eqa in eqa_ls:
    #                 eqa_tempdir = get_objpixmap(self.paper.pdf, eqa)
    #                 topic_search(self, name).addSubTopicbyImage(
    #                     eqa_tempdir, eqa_ls.index(eqa))

        # Section Dict Extract

    def get_section_titles(self, withlevel=False, verbose=False):
        section_title = []
        self.pdf = fitz.open(self.path)
        # ref_break_flag = False
        level1_matchstr = SECTION_TITLE_MATCHSTR[0]
        level2_matchstr = SECTION_TITLE_MATCHSTR[1]
        for page in self.pdf:
            blocks = page.get_text("dict", flags=0)["blocks"]
            for block in blocks:
                # Assume: Section title is the first "Line" or multiple "Lines" that have the same y position in one "block"
                is_equation = False
                lines = block["lines"]
                pos_y = block["lines"][0]["bbox"][1]
                tol = 1
                line_text = ""
                for line in lines:
                    for span in line["spans"]:
                        if span['font'].startswith('CM') or span['font'].startswith('MSBM'):
                            is_equation = True
                            break
                    if abs(pos_y - line["bbox"][1]) < tol:
                        line_text = line_text + "".join([span["text"] for span in line["spans"]]) + "\n"
                    else:
                        break
                if is_inbox(block['bbox'][0:2], get_bounding_box(getColumnRectLegacy(page, ymargin=40))) \
                        and is_inbox(block['bbox'][2:4],
                                     get_bounding_box(getColumnRectLegacy(page, ymargin=40))) and not is_equation:
                    if re.match(level2_matchstr, line_text):
                        if line_text.startswith('I.') and len(section_title) < 7:  # Considering I. i.e. ABCDEFGHI
                            section_title.append((line_text, 1))
                        else:
                            section_title.append((line_text, 2))
                    elif re.match(level1_matchstr, line_text):
                        section_title.append((line_text, 1))
                # if re.match(REF_MATCHSTR, line_text):
                #     ref_break_flag = True
                #     break
            #     if ref_break_flag:
            #         break
            # if ref_break_flag:
            #     break
        section_title = [(re.search(ABS_MATCHSTR, self.all_text).group(), 1)] \
                        + section_title \
                        + [(re.search(REF_MATCHSTR, self.all_text).group(), 1)]
        return section_title if withlevel else [t[0] for t in section_title]

    def get_section_textposdict(self):
        section_title = self.get_section_titles()
        section_dict = {}
        for i in range(0, len(section_title) - 1):
            title = section_title[i]
            latter_title = section_title[i + 1]
            begin_pos = self.all_text.find(title)
            end_pos = self.all_text.find(latter_title)
            section_dict[title] = (begin_pos, end_pos)
            if begin_pos == -1:
                print(f"Warning: {title} not found in all_text.")
        return section_dict

    # def get_section_textdict(self, remove_title=False):
    #     """
    #     Get section text dict of the paper.
    #     :return: Dict of section titles with text content
    #     FIXME: This will not get Reference content
    #     """
    #     section_title = self.get_section_titles(withlevel=False)
    #     pos_dict = self.get_section_textposdict()
    #     section_dict = {}
    #     for title in section_title[:-1]:
    #         if not remove_title:
    #             section_dict[title] = self.all_text[pos_dict[title][0]:pos_dict[title][1]]
    #         else:
    #             section_dict[title] = self.all_text[pos_dict[title][0] + len(title):pos_dict[title][1]]
    #     return section_dict

    # def get_section_equationdict(self, legacy=False):
    #     """
    #     Get equation dict of each section
    #
    #     :param legacy: True for legacy equation extraction method
    #     :return: Dict of section titles with tuple item list
    #     (equation_text_pos, page_number, equation_bbox)
    #     """
    #     eqa_ls = []
    #     for i in range(len(self.pdf)):
    #         page = self.pdf[i]
    #         if legacy:
    #             eq_box = get_eqbox(page)
    #         else:
    #             eq_box = getEqRect(page)
    #         if eq_box:
    #             for box in eq_box:
    #                 eqa_ls.append((get_box_textpos(page, box, self.all_text),
    #                                i, box))
    #     section_title = self.get_section_titles()
    #     pos_dict = self.get_section_textposdict()
    #     section_dict = {}
    #     for title in section_title[:-1]:
    #         section_dict[title] = []
    #         for eqa in eqa_ls:
    #             if eqa[0] > pos_dict[title][0] and eqa[0] <= pos_dict[title][1]:
    #                 section_dict[title].append(eqa)
    #             elif section_dict[title]:
    #                 break
    #     return section_dict

    def get_section_imagedict_default(self, verbose=False, snap_with_caption=True):
        """
        Get image dict of each section

        :return: Dict of section titles with tuple item list
        (img_text_pos, page_number, img_bbox)
        """
        img_ls = []
        for i in range(len(self.pdf)):
            page = self.pdf[i]
            img_box = getFigRect(page)
            if img_box:
                for box in img_box:
                    img_ls.append((get_box_textpos(page, box, self.all_text),
                                   i, box))
        if verbose:
            logger.debug(f'Total images found: {str(len(img_ls))}')
        section_title = self.get_section_titles()
        pos_dict = self.get_section_textposdict()
        section_dict = {}
        match_cnt = 0
        for title in section_title[:-1]:
            section_dict[title] = []
            for img in img_ls:
                if img[0] > pos_dict[title][0] and img[0] < pos_dict[title][1]:
                    section_dict[title].append(img)
                    match_cnt += 1
                # elif section_dict[title]:
                #     break
        if verbose:
            logger.debug('Images match the content: f{match_cnt}')
        return section_dict

    def get_section_imagedict_jvm(self, snap_with_caption=True, verbose=False):
        """
        Get image dict of each section
        :return: Dict of section titles with tuple item list
        (img_text_pos, page_number, img_bbox, img_caption)
        """
        self.pdf = fitz.open(self.path)  # pdf文档
        img_ls = []
        for d in self.PDFF2data.get('figures', []):
            if snap_with_caption:
                box = get_bounding_box([
                    (d['regionBoundary']['x1'], d['regionBoundary']['y1'],
                     d['regionBoundary']['x2'], d['regionBoundary']['y2']),
                    (d['captionBoundary']['x1'], d['captionBoundary']['y1'],
                     d['captionBoundary']['x2'], d['captionBoundary']['y2'])])
                img_ls.append((get_box_textpos(self.pdf[d['page']], box, self.all_text),
                               d['page'], box))
            else:
                box = (d['regionBoundary']['x1'], d['regionBoundary']['y1'],
                       d['regionBoundary']['x2'], d['regionBoundary']['y2'])
                img_ls.append((get_box_textpos(self.pdf[d['page']], box, self.all_text),
                               d['page'], box, d['caption']))
        if verbose:
            logger.debug(f'Total images found: {str(len(img_ls))}')
        section_title = self.get_section_titles()
        pos_dict = self.get_section_textposdict()
        section_dict = {}
        match_cnt = 0
        for title in section_title[:-1]:
            section_dict[title] = []
            # if verbose:
            #     print('Title:', title, 'Begin pos:',
            #           pos_dict[title][0], 'End pos:', pos_dict[title][1])
            for img in img_ls:
                if img[0] > pos_dict[title][0] and img[0] < pos_dict[title][1]:
                    section_dict[title].append(img)
                    match_cnt += 1
                # elif section_dict[title]:
                #     break
        # if verbose:
        #     print('Images match the content:', match_cnt)
        return section_dict


def main():
    path = r'W:\Personal_Project\metaunitech\arxiv_daily\demo2.pdf'
    paper = Paper(path=path)
    paper.parse_pdf()
    paper.get_chapter_names()
    paper.get_image_path()
    for key, value in paper.section_text_dict.items():
        logger.debug(f"{key}, {value}")
        logger.debug("*" * 40)


if __name__ == '__main__':
    main()

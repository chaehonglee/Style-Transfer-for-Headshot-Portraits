from style_transfer import StyleTransfer

if __name__ == '__main__':
    ST = StyleTransfer('input/style1.jpg',
                       'input/face.jpg',
                       'input/mask_style1.jpg',
                       'input/mask_face.jpg')
    ST.run()

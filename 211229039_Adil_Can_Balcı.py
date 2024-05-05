import tkinter as tk
import math
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageFilter, ImageTk
import numpy as np  
import cv2
import pandas as pd
import os 
from os.path import expanduser


def resize_with_average_interpolation(image, size):
    new_width, new_height = size
    width, height = image.size
    new_image = Image.new("RGB", (new_width, new_height))
    for y in range(new_height):
        for x in range(new_width):
            orig_x = x * width // new_width
            orig_y = y * height // new_height
            r_total = g_total = b_total = 0
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    xx = min(max(0, orig_x + dx), width - 1)
                    yy = min(max(0, orig_y + dy), height - 1)
                    r, g, b = image.getpixel((xx, yy))
                    r_total += r
                    g_total += g
                    b_total += b
            r_avg = r_total // 9
            g_avg = g_total // 9
            b_avg = b_total // 9
            new_image.putpixel((x, y), (r_avg, g_avg, b_avg))
    return new_image


class ImageProcessingForm(tk.Toplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.title("Görüntü İşleme Penceresi")
        self.geometry("1024x768")
        self.grab_set() 
        self.btnLoadImage = tk.Button(self, text="Görüntü Yükle", command=self.load_image)
        self.btnLoadImage.place(x=10, y=10, width=100, height=30)
        self.btnBlur = tk.Button(self, text="Bulanıklaştır", command=self.blur_image)
        self.btnBlur.place(x=120, y=10, width=100, height=30)
        self.btnBlur.config(state="disabled")
        self.btnRegionBlur = tk.Button(self, text="Bölgesel Bulanıklaştır", command=self.enable_region_blur)
        self.btnRegionBlur.place(x=230, y=10, width=150, height=30)
        self.canvas = tk.Canvas(self, bg="white")
        self.canvas.place(x=10, y=40, width=1900, height=1600)
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.selected_region = None
        self.selection_rectangle = None
        self.is_region_blur_enabled = False
        self.is_mouse_down = False
        self.last_location = None
        

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Resim Dosyaları", "*.jpg;*.jpeg;*.png;*.bmp")])
        if file_path:
            try:
                self.image = Image.open(file_path)
                self.show_image()
                self.btnBlur.config(state="normal")
            except Exception as e:
                messagebox.showerror("Hata", "Görüntü yüklenirken bir hata oluştu: " + str(e))

    def show_image(self):
        self.photo = ImageTk.PhotoImage(self.image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

        

    def blur_image(self):
        if self.image:
            blurred_image = self.image.filter(ImageFilter.BLUR)
            self.image = blurred_image
            self.show_image()
        else:
            messagebox.showwarning("Uyarı", "Önce bir görüntü yükleyin!")

    def enable_region_blur(self):
        self.is_region_blur_enabled = True
        self.selection_rectangle = None

    def on_mouse_down(self, event):
        if self.is_region_blur_enabled:
            self.is_mouse_down = True
            self.last_location = (event.x, event.y)
            self.selected_region = (event.x, event.y, event.x, event.y)

    def on_mouse_move(self, event):
        if self.is_mouse_down:
            if self.image:
                self.selection_rectangle = (self.selected_region[0], self.selected_region[1], event.x, event.y)
                self.show_selection_rectangle()

    def on_mouse_up(self, event):
        if self.is_mouse_down:
            self.is_mouse_down = False
            self.is_region_blur_enabled = False
            if self.selection_rectangle:
                x1, y1, x2, y2 = self.selection_rectangle
                region = self.image.crop((x1, y1, x2, y2))
                blurred_region = region.filter(ImageFilter.BLUR)
                self.image.paste(blurred_region, (x1, y1, x2, y2))
                self.show_image()

    def show_selection_rectangle(self):
        if self.selection_rectangle:
            x1, y1, x2, y2 = self.selection_rectangle
            self.canvas.delete("selection_rectangle")
            self.canvas.create_rectangle(x1, y1, x2, y2, outline="red", tag="selection_rectangle")

    
class MainForm(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Dijital Görüntü İşleme Dersi")
        self.geometry("800x600")
        self.lblTitle = tk.Label(self, text="Dijital Görüntü İşleme Dersi", font=("Arial", 12, "bold"))
        self.lblTitle.place(x=50, y=50)
        self.lblStudentInfo = tk.Label(self, text="Öğrenci No: 211229039\nAdı Soyadı: Adil Can Balcı")
        self.lblStudentInfo.place(x=50, y=100)
        self.menu = tk.Menu(self)
        self.assignments_menu = tk.Menu(self.menu, tearoff=0)
        self.assignments_menu.add_command(label="Ödev Ekle", command=self.add_assignment)
        assignments = {
             1: "Ödev 1: Temel İşlevsellik Oluştur",
             2: "Ödev 2: Temel Görüntü Operasyonları ve İnterpolasyon",
             3: "Vize Ödevi",
             4: "Ödev 4",
             # Diğer Ödevler....
        }

        for i, assignment_name in assignments.items():
            self.assignments_menu.add_command(label=assignment_name, command=lambda i=i: self.show_assignment(i))


        self.menu.add_cascade(label="Ödevler", menu=self.assignments_menu)
        self.config(menu=self.menu)
        self.is_mdi_container = True

    def add_assignment(self):
        new_assignment_name = simpledialog.askstring("Ödev Ekle", "Yeni ödev ismini girin:")
        if new_assignment_name:
            self.assignments_menu.add_command(label=new_assignment_name, command=self.show_custom_assignment)
            messagebox.showinfo("Bilgi", "Yeni bir ödev eklendi.")

    def show_assignment(self, assignment_number):
        if assignment_number == 1:
            self.image_processing_form = ImageProcessingForm(self)
        elif assignment_number == 2:
            self.image_processing_form2 = ImageProcessingForm2(self)
        elif assignment_number == 3:
            self.image_processing_form3 = ImageProcessingForm3(self)
        else:
            messagebox.showinfo("Ödev", f"Ödev {assignment_number} seçildi.")

    def show_custom_assignment(self):
        messagebox.showinfo("Ödev", "Seçilen ödev açıldı.")


class ImageProcessingForm2(ImageProcessingForm):
    def __init__(self, master=None):
        super().__init__(master)
        self.image = None  
        self.btnEnlarge = tk.Button(self, text="Büyüt", command=self.enlarge_image)
        self.btnEnlarge.place(x=120, y=10, width=70, height=30)
        self.btnShrink = tk.Button(self, text="Küçült", command=self.shrink_image)
        self.btnShrink.place(x=200, y=10, width=70, height=30)
        self.btnZoomIn = tk.Button(self, text="Yakınlaştır", command=lambda: self.zoom_in(2))
        self.btnZoomIn.place(x=280, y=10, width=90, height=30)
        self.btnZoomOut = tk.Button(self, text="Uzaklaştır", command=lambda: self.zoom_out(2))
        self.btnZoomOut.place(x=380, y=10, width=90, height=30)
        self.btnRotate = tk.Button(self, text="Döndür", command=lambda: self.rotate_image(90)) 
        self.btnRotate.place(x=480, y=10, width=70, height=30)
        self.interpolation_method = tk.StringVar(self)
        self.interpolation_method.set("Bilinear")  
        self.interpolation_menu = tk.OptionMenu(self, self.interpolation_method, "Bilinear", "Bicubic", "Average")
        self.interpolation_menu.place(x=580, y=15, width=100, height=25)
        self.btnApplyInterpolation = tk.Button(self, text="Uygula", command=self.apply_selected_interpolation)
        self.btnApplyInterpolation.place(x=690, y=10, width=70, height=30)
        self.btnLoadImage.place(x=10, y=10, width=100, height=30)
        self.btnBlur.place_forget()
        self.btnRegionBlur.place_forget()

    def enlarge_image(self):
        percent_str = simpledialog.askstring("Büyütme Yüzdesi", "Büyütme yüzdesini girin:")
        try:
            percent = float(percent_str)
            if percent <= 0:
                messagebox.showwarning("Uyarı", "Yüzde değeri pozitif bir sayı olmalıdır.")
                return
        except ValueError:
            messagebox.showwarning("Uyarı", "Geçersiz yüzde değeri.")
            return

        if self.image:
            factor = 1 + (percent / 100)  
            width, height = self.image.size
            new_width = int(width * factor)
            new_height = int(height * factor)
            enlarged_image = self.image.resize((new_width, new_height), resample=Image.BICUBIC)
            self.image = enlarged_image
            self.show_image()
        else:
            messagebox.showwarning("Uyarı", "Önce bir görüntü yükleyin!")

    def shrink_image(self):
        percent_str = simpledialog.askstring("Küçültme Yüzdesi", "Küçültme yüzdesini girin:")
        try:
            percent = float(percent_str)
            if percent <= 0:
                messagebox.showwarning("Uyarı", "Yüzde değeri pozitif bir sayı olmalıdır.")
                return
        except ValueError:
            messagebox.showwarning("Uyarı", "Geçersiz yüzde değeri.")
            return

        if self.image:
            factor = 1 - (percent / 100) 
            width, height = self.image.size
            new_width = int(width * factor)
            new_height = int(height * factor)
            shrunken_image = self.image.resize((new_width, new_height), resample=Image.BILINEAR)
            self.image = shrunken_image
            self.show_image()
        else:
            messagebox.showwarning("Uyarı", "Önce bir görüntü yükleyin!")

    def zoom_in(self, factor):
        if self.image:
            width, height = self.image.size
            new_width = int(width * factor)
            new_height = int(height * factor)
            zoomed_image = self.image.resize((new_width, new_height), Image.BICUBIC)
            self.image = zoomed_image
            self.show_image()
        else:
            messagebox.showwarning("Uyarı", "Önce bir görüntü yükleyin!")

    def zoom_out(self, factor):
        if self.image:
            width, height = self.image.size
            new_width = int(width / factor)
            new_height = int(height / factor)
            zoomed_image = self.image.resize((new_width, new_height), Image.BILINEAR)
            self.image = zoomed_image
            self.show_image()
        else:
            messagebox.showwarning("Uyarı", "Önce bir görüntü yükleyin!")

    def rotate_image(self, angle=90):
        angle_str = simpledialog.askstring("Döndürme Açısı", "Döndürme açısını girin (derece):")
        try:
         angle = float(angle_str)
        except ValueError:
         messagebox.showwarning("Uyarı", "Geçersiz açı değeri.")
         return

        if self.image:
            rotated_image = self.image.rotate(angle, expand=True)
            self.image = rotated_image
            self.show_image()
        else:
         messagebox.showwarning("Uyarı", "Önce bir görüntü yükleyin!")


    def apply_selected_interpolation(self):
        interpolation_method = self.interpolation_method.get()
        if self.image:
            if interpolation_method == "Bilinear":
                interpolated_image = self.image.resize(self.image.size, Image.BILINEAR)
            elif interpolation_method == "Bicubic":
                interpolated_image = self.image.resize(self.image.size, Image.BICUBIC)
            elif interpolation_method == "Average":
                interpolated_image = resize_with_average_interpolation(self.image, self.image.size)
            else:
                messagebox.showwarning("Uyarı", "Geçersiz interpolasyon yöntemi!")
                return
            self.image = interpolated_image
            self.show_image()
        else:
            messagebox.showwarning("Uyarı", "Önce bir görüntü yükleyin!")


class ImageProcessingForm3(ImageProcessingForm):
    def __init__(self, master=None):
        super().__init__(master)
        self.image = None
        self.btnLoadImage.place(x=10, y=10, width=100, height=30)
        self.btnSoru1 = tk.Button(self, text="Soru 1", command=self.soru_1_function)
        self.btnSoru1.place(x=120, y=10, width=100, height=30)

        self.btnSoru2 = tk.Button(self, text="Yol", command=self.soru_2_yol_function)
        self.btnSoru2.place(x=230, y=10, width=100, height=30)

        self.btnSoru5 = tk.Button(self, text="Göz", command=self.soru_2_goz_function)
        self.btnSoru5.place(x=350, y=10, width=100, height=30)

        self.btnSoru3 = tk.Button(self, text="Soru 3", command=self.soru_3_function)
        self.btnSoru3.place(x=470, y=10, width=100, height=30)

        self.btnSoru4 = tk.Button(self, text="Soru 4", command=self.soru_4_function)
        self.btnSoru4.place(x=600, y=10, width=100, height=30)

        self.btnBlur.place_forget()  
        self.btnRegionBlur.place_forget()  

    def soru_1_function(self):
        if self.image:
            def standard_sigmoid(x):
                return 255 * (1 / (1 + math.exp(-x / 255)))
            def horizontal_shift_sigmoid(x):
                return 255 * (1 / (1 + math.exp(-(x - 128) / 32)))
            def slope_sigmoid(x):
                return 255 * (1 / (1 + math.exp(-0.1 * (x - 128))))
            def custom_function(x):
                return 255 * (0.5 + 0.5 * math.sin((x - 128) / 128 * math.pi))
            image_array = np.array(self.image)
            enhanced_image_array = np.zeros_like(image_array, dtype=float)
            enhanced_images = []
            for sigmoid_func in [standard_sigmoid, horizontal_shift_sigmoid, slope_sigmoid, custom_function]:
                for i in range(3):
                    enhanced_image_array[:, :, i] = np.vectorize(sigmoid_func)(image_array[:, :, i])
                enhanced_image = Image.fromarray(enhanced_image_array.astype(np.uint8))
                enhanced_images.append(enhanced_image)
            self.show_sigmoid_images(enhanced_images)
        else:
            messagebox.showwarning("Uyarı", "Önce bir görüntü yükleyin!")

    def show_sigmoid_images(self, images):
        sigmoid_window = tk.Toplevel(self)
        sigmoid_window.title("Sigmoid Fonksiyonları ile Güçlendirilmiş Görüntüler")
        for i, image in enumerate(images):
            photo = ImageTk.PhotoImage(image)
            label = tk.Label(sigmoid_window, image=photo)
            label.image = photo  # Referansı saklamak için gerekli
            label.grid(row=0, column=i, padx=10, pady=10)

    def soru_2_yol_function(self):
        if self.image:
            gray_image = cv2.cvtColor(np.array(self.image), cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 10000, threshold=2, minLineLength=180, maxLineGap=8)
            image_with_lines = np.array(self.image).copy()
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(image_with_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)
                img_with_lines = Image.fromarray(image_with_lines)
                self.canvas.delete("all")  
                self.image_tk = ImageTk.PhotoImage(img_with_lines)
                self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image_tk)
            else:
                messagebox.showinfo("Bilgi", "Çizgi bulunamadı!")
        else:
            messagebox.showwarning("Uyarı", "Önce bir görüntü yükleyin!")


    def soru_2_goz_function(self):
        if self.image:
            img = cv2.cvtColor(np.array(self.image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                                    param1=200, param2=30, minRadius=10, maxRadius=30)
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                for (x, y, r) in circles:
                    cv2.circle(img, (x, y), r, (0, 255, 0), 4)
                cv2.imshow("Goz Tespiti", img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        else:
            messagebox.showwarning("Uyarı", "Önce bir görüntü yükleyin!")


    def soru_3_function(self):
        if self.image:
            image_array = np.array(self.image)
            edges = cv2.Canny(image_array, 50, 150, apertureSize=3)
            psf = self.estimate_psf(edges, image_array.shape[:2])
            deblurred_image = self.deblur_image(image_array, psf)
            self.show_deblurred_images(image_array, deblurred_image)
        else:
            messagebox.showwarning("Uyarı", "Önce bir görüntü yükleyin!")

    def estimate_psf(self, image, image_shape):
        psf = cv2.blur(image, (5, 5))  
        psf_resized = cv2.resize(psf, (image_shape[1], image_shape[0]))
        return psf_resized

    def deblur_image(self, image, psf):
        deblurred_image = cv2.filter2D(image, -1, psf)  
        return deblurred_image

    def show_deblurred_images(self, original_image, deblurred_image):
        deblurred_window = tk.Toplevel(self)
        deblurred_window.title("Deblurred Images")
        original_photo = ImageTk.PhotoImage(image=Image.fromarray(original_image))
        deblurred_photo = ImageTk.PhotoImage(image=Image.fromarray(deblurred_image))
        original_label = tk.Label(deblurred_window, image=original_photo)
        original_label.image = original_photo
        original_label.grid(row=0, column=0, padx=10, pady=10)
        deblurred_label = tk.Label(deblurred_window, image=deblurred_photo)
        deblurred_label.image = deblurred_photo
        deblurred_label.grid(row=0, column=1, padx=10, pady=10)


    def soru_4_function(self):
        if self.image:
            image_np = np.array(self.image)
            hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)  
            lower_green = np.array([35, 100, 100])  
            upper_green = np.array([85, 255, 255])  
            mask = cv2.inRange(hsv, lower_green, upper_green)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            df = pd.DataFrame(columns=["No", "Center", "Length", "Width", "Diagonal", "Energy", "Entropy", "Mean", "Median"])
            for i, contour in enumerate(contours):
                x, y, w, h = cv2.boundingRect(contour)
                center = (x + w // 2, y + h // 2)
                diagonal = np.sqrt(w**2 + h**2)
                roi = image_np[y:y+h, x:x+w]  
                if roi.size == 0:
                    continue
                energy = np.sum(roi.astype(np.float32)**2)
                entropy = -np.sum((roi / 255) * np.log2(roi / 255 + 1e-10))
                mean = np.mean(roi)
                median = np.median(roi)
                df.loc[i] = [i+1, center, w, h, diagonal, energy, entropy, mean, median]
            desktop_path = expanduser("~") + "/Desktop"
            excel_file_path = os.path.join(desktop_path, "green_areas.xlsx")
            df.to_excel(excel_file_path, index=False)
            print("Veriler Excel dosyasına masaüstüne kaydedildi:", excel_file_path)
            return df
        else:
            print("Uyarı: Önce bir görüntü yükleyin!")
            return None



if __name__ == "__main__":
    app = MainForm()
    app.mainloop()

import flet

def main(page):
    
    def load_agent_model(zip_path, extract_path):
        if not os.path.exists(extract_path):
            os.makedirs(extract_path)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        
        model_path = os.path.join(extract_path, 'agent_model.pth')  # Adjust the filename if different
        model = torch.load(model_path, map_location=torch.device('cpu'))
        model.eval()
        return model

    # Function to show the performance page
    def show_performance(page):
        zip_path = "/Users/henryforrest/Documents/Computer Science Work/Dissertation/flet/Dissertation/agent.zip"  # The model zip file stored in the same directory
        extract_path = "agent_model"  # Extracted folder
        
        try:
            model = load_agent_model(zip_path, extract_path)
            message = "Model loaded successfully!"
            color = ft.colors.GREEN
        except Exception as e:
            message = f"Error loading model: {str(e)}"
            color = ft.colors.RED
            model = None
        
        def predict_performance(e):
            if model is None:
                result_text.value = "Model not loaded. Cannot make predictions."
                result_text.color = ft.colors.RED
            else:
                # Example input: Modify this based on actual input requirements
                dummy_input = torch.randn(1, 10)  # Adjust the shape to match your model's input
                prediction = model(dummy_input).detach().numpy()
                result_text.value = f"Predicted performance: {prediction}"
                result_text.color = ft.colors.BLUE
            page.update()
        
        result_text = ft.Text(value="", size=20)
        
        performance_page = ft.Container(
            content=ft.Column(
                controls=[
                    ft.Text("Performance Analysis", size=40, weight="bold", color="#1e3a8a"),
                    ft.Text(message, color=color, size=20),
                    ft.ElevatedButton("Predict Performance", on_click=predict_performance),
                    result_text,
                ],
                alignment=ft.MainAxisAlignment.CENTER,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            height=page.height,
            width=page.width,
            bgcolor="#ffffff",
            padding=ft.padding.all(20),
        )
        
        page.controls.clear()
        page.add(navbar, performance_page)
        page.update()
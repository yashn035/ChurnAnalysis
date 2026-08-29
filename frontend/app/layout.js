import "./globals.css";

export const metadata = {
  title: "Customer Churn Predictor | ML Dashboard",
  description: "Real-time subscriber retention risk analytics powered by FastAPI & Next.js",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en" className="h-full antialiased dark">
      <head>
        <script src="https://cdn.tailwindcss.com"></script>
      </head>
      <body className="min-h-full flex flex-col bg-[#0b0f19] text-[#f3f4f6]">{children}</body>
    </html>
  );
}

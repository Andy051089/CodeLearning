package hello;
import java.util.Scanner;
public class WeatherHomeOut {
	public static void main(String[] args) {
		Scanner input = new Scanner(System.in);
		System.out.print("請輸入天氣是晴還是陰，晴打1，陰打2");
		int ans = input.nextInt();
		if(ans==1) {
			System.out.println("今天氣晴可以去1.逛街2.散步");
			int ans1 = input.nextInt();
			if(ans1==1) {
				System.out.println("今天天氣晴去逛街");
			}else {
				System.out.println("今天天氣晴去散步");
			}
		}else {
			System.out.println("今天天氣陰可以在家1.看電影2.打掃");
			int ans2 = input.nextInt();
			if(ans2==1) {
				System.out.println("今天天氣陰在家看電影");
			}else {
				System.out.println("今天天氣陰在家打掃");
			}
		}
	}
}
